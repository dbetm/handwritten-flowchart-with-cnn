# -*- coding: utf-8 -*-

from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
import threading
import itertools
import traceback

from . data_augment import DataAugment
from . utilities.image_tools import ImageTools


class Metrics(object):
	"""Methods for object-recogntion metrics"""

	def __init__(self):
		super(Metrics, self).__init__()

	@staticmethod
	def union(au, bu, area_intersection):
		"""Calculate total area between two rectangles."""

		area_a = (au[2]-au[0]) * (au[3]-au[1])
		area_b = (bu[2]-bu[0]) * (bu[3]-bu[1])
		area_union = area_a + area_b - area_intersection
		return area_union

	@staticmethod
	def intersection(ai, bi):
		"""Calculate shared total area between two rectangles."""

		x = max(ai[0], bi[0])
		y = max(ai[1], bi[1])
		w = min(ai[2], bi[2]) - x
		h = min(ai[3], bi[3]) - y

		if w < 0 or h < 0:
			return 0
		return w * h

	@staticmethod
	def iou(a, b):
		"""Calculate metric Intersection over Union."""

		# a and b should be (x1,y1,x2,y2)
		if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
			return 0.0

		area_i = Metrics.intersection(a, b)
		area_u = Metrics.union(a, b, area_i)

		return float(area_i) / float(area_u+1e-6)


class SampleSelector:
	"""Selector of instances."""

	def __init__(self, class_count):
		# ignore classes that have zero samples
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		# Make an copy of the images
		self.class_cycle = itertools.cycle(self.classes)

		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):
		"""Skip an instance to balance the number of classes."""

		class_in_img = False

		for bbox in img_data['bboxes']:
			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


class Threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""

	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		"""Recover next item from the iterator/generator."""

		# The 'with' statement clarifies code that
		# previously would use try...finally blocks
		with self.lock:
			return next(self.it)


class Utils(object):
	"""Class with methods for data generators"""

	def __init__(self):
		super(Utils, self).__init__()

	@staticmethod
	def threadsafe_generator(f):
		"""A decorator that takes a generator function and makes it thread-safe.
		"""
		def g(*a, **kw):
			return Threadsafe_iter(f(*a, **kw))
		return g

	@staticmethod
	def calc_rpn(
			config,
			data,
			width,
			height,
			new_width,
			new_height,
			img_len_calc_func):
		"""Calculate Region Proposals using the anchors."""

		downscale = float(config.rpn_stride)
		anchor_sizes = config.anchor_box_scales
		anchor_ratios = config.anchor_box_ratios
		num_anchors = len(anchor_sizes) * len(anchor_ratios)

		# calculate the output map size based on the network architecture
		(output_width, output_height) = img_len_calc_func(new_width, new_height)
		n_anchratios = len(anchor_ratios)

		# initialise empty output objectives
		y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
		y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
		y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

		# Get number of bounding boxes
		num_bboxes = len(data['bboxes'])

		# initialise anchors-vectors
		num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
		best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
		best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
		best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
		best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

		gta = Utils.get_GT_box_coordinates(
			data,
			num_bboxes,
			width,
			height,
			new_width,
			new_height
		)

		ans = Utils.get_RPN_ground_truth(
			config, anchor_sizes, anchor_ratios, n_anchratios, output_width,
			output_height, new_width, new_height, downscale, num_bboxes, gta,
			data, y_rpn_overlap, y_is_box_valid, y_rpn_regr, num_anchors_for_bbox,
			best_anchor_for_bbox, best_iou_for_bbox, best_x_for_bbox,
			best_dx_for_bbox
		)
		y_rpn_overlap, y_is_box_valid, y_rpn_regr = ans[0]
		num_anchors_for_bbox, best_anchor_for_bbox, best_iou_for_bbox, best_x_for_bbox, best_dx_for_bbox = ans[1]
		# We ensure that every bbox has at least one positive RPN region
		ans = Utils.ensure_least_one_pos_RPN(
			y_rpn_overlap, y_is_box_valid, y_rpn_regr,
			num_anchors_for_bbox, best_anchor_for_bbox, n_anchratios,
			best_dx_for_bbox
		)
		# Recovery items of list
		y_rpn_overlap, y_is_box_valid, y_rpn_regr = ans

		# Select only positive and valid regions
		pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1,
								y_is_box_valid[0, :, :, :] == 1)
							)
		# Select only positive and valid backgrounds
		neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0,
								y_is_box_valid[0, :, :, :] == 1)
							)
		# Number of positives
		num_pos = len(pos_locs[0])

		# One issue is that the RPN has many more negative than positive regions,
		# so we turn off some of the negative regions. We also limit it to 256 regions.
		num_regions = 256

		if len(pos_locs[0]) > num_regions / 2:
			val_locs = random.sample(
				range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2
			)
			aux1 = pos_locs[0][val_locs]
			aux2 = pos_locs[1][val_locs]
			y_is_box_valid[0, aux1, aux2, pos_locs[2][val_locs]] = 0
			num_pos = num_regions / 2

		if len(neg_locs[0]) + num_pos > num_regions:
			val_locs = random.sample(
				range(len(neg_locs[0])), len(neg_locs[0]) - num_pos
			)
			aux1 = neg_locs[0][val_locs]
			aux2 = neg_locs[1][val_locs]
			y_is_box_valid[0, aux1, aux2, neg_locs[2][val_locs]] = 0

		y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
		y_rpn_regr = np.concatenate(
			[np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr],
			axis=1
		)

		return np.copy(y_rpn_cls), np.copy(y_rpn_regr)

	@staticmethod
	def get_GT_box_coordinates(
			data,
			num_bboxes,
			width,
			height,
			new_width,
			new_height):
		"""Get the GT box coordinates, and resize to account for
		image resizing.
		"""

		gta = np.zeros((num_bboxes, 4))
		for bbox_num, bbox in enumerate(data['bboxes']):
			# get the GT box coordinates, and resize to account
			# for image resizing
			gta[bbox_num, 0] = bbox['x1'] * (new_width/float(width))
			gta[bbox_num, 1] = bbox['x2'] * (new_width/float(width))
			gta[bbox_num, 2] = bbox['y1'] * (new_height/float(height))
			gta[bbox_num, 3] = bbox['y2'] * (new_height/float(height))

		return gta

	@staticmethod
	def get_RPN_ground_truth(
			config, anchor_sizes, anchor_ratios, n_anchratios,
			output_width, output_height, new_width, new_height, downscale,
			num_bboxes, gta, data, y_rpn_overlap, y_is_box_valid, y_rpn_regr,
			num_anchors_for_bbox, best_anchor_for_bbox, best_iou_for_bbox,
			best_x_for_bbox, best_dx_for_bbox):
		"""RPN Ground Truth"""

		for anchor_size_idx in range(len(anchor_sizes)):
			for anchor_ratio_idx in range(n_anchratios):
				anchor_x = anchor_sizes[anchor_size_idx]
				anchor_x *= anchor_ratios[anchor_ratio_idx][0]
				anchor_y = anchor_sizes[anchor_size_idx]
				anchor_y *= anchor_ratios[anchor_ratio_idx][1]

				for ix in range(output_width):
					# x-coordinates of the current anchor box
					x1_anc = downscale * (ix+0.5) - anchor_x / 2
					x2_anc = downscale * (ix+0.5) + anchor_x / 2
					# ignore boxes that go across image boundaries
					if x1_anc < 0 or x2_anc > new_width:
						continue
					for jy in range(output_height):
						# y-coordinates of the current anchor box
						y1_anc = downscale * (jy+0.5) - anchor_y / 2
						y2_anc = downscale * (jy+0.5) + anchor_y / 2
						# ignore boxes that go across image boundaries
						if y1_anc < 0 or y2_anc > new_height:
							continue
						# bbox_type indicates an anchor probably target
						# default isn't target (negative)
						bbox_type = 'neg'
						""" This is the best IOU for the (x,y) coord and the
						current anchor. Note that this is different from the
						best IOU for a GT bbox.
						"""
						best_iou_for_loc = 0.0
						for bbox_num in range(num_bboxes):
							# get IOU of the current GT box and the current anchor box
							curr_iou = Metrics.iou(
											[gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
											[x1_anc, y1_anc, x2_anc, y2_anc]
										)
							# calculate the regression targets if they will be needed
							cond1 = curr_iou > best_iou_for_bbox[bbox_num]
							cond2 = curr_iou > config.rpn_max_overlap
							if cond1 or cond2:
								cx = (gta[bbox_num, 0]+gta[bbox_num, 1]) / 2.0
								cy = (gta[bbox_num, 2]+gta[bbox_num, 3]) / 2.0
								cxa = (x1_anc+x2_anc) / 2.0
								cya = (y1_anc+y2_anc) / 2.0

								tx = (cx-cxa) / (x2_anc-x1_anc)
								ty = (cy-cya) / (y2_anc-y1_anc)
								div = x2_anc - x1_anc
								tw = np.log((gta[bbox_num, 1]-gta[bbox_num, 0]) / (div))
								div = y2_anc - y1_anc
								th = np.log((gta[bbox_num, 3]-gta[bbox_num, 2]) / (div))

							if data['bboxes'][bbox_num]['class'] != 'bg':
								""" all GT boxes should be mapped to an anchor
								box, so we keep track of which anchor box was best
								"""
								if curr_iou > best_iou_for_bbox[bbox_num]:
									best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]

									best_iou_for_bbox[bbox_num] = curr_iou
									best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
									best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]
								""" We set the anchor to positive if the IOU is > 0.7.
								It doesn't matter if there was another better box,
								it just indicates overlap.
								"""
								if curr_iou > config.rpn_max_overlap:
									bbox_type = 'pos'
									num_anchors_for_bbox[bbox_num] += 1
									"""We update the regression layer target if
									this IoU is the best for the current (x,y)
									and anchor position.
									"""
									if curr_iou > best_iou_for_loc:
										best_iou_for_loc = curr_iou
										best_regr = (tx, ty, tw, th)

								""" If the IoU is > 0.3 and < 0.7, it is
								ambiguous and no included in the objective.
								"""
								if config.rpn_min_overlap < curr_iou < config.rpn_max_overlap:
									# gray zone between neg and pos
									if bbox_type != 'pos':
										bbox_type = 'neutral'
						cond1 = bbox_type == 'neg' or bbox_type == 'pos'
						cond2 = bbox_type == 'pos'
						# turn on or off outputs depending on IoUs
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = cond1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = cond2
						if bbox_type == 'pos':
							start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
							y_rpn_regr[jy, ix, start:start+4] = best_regr
		aux1 = [y_rpn_overlap, y_is_box_valid, y_rpn_regr]
		aux2 = [
			num_anchors_for_bbox,
			best_anchor_for_bbox,
			best_iou_for_bbox,
			best_x_for_bbox,
			best_dx_for_bbox
		]

		return [aux1, aux2]

	@staticmethod
	def ensure_least_one_pos_RPN(
			y_rpn_overlap, y_is_box_valid, y_rpn_regr,
			num_anchors_for_bbox, best_anchor_for_bbox, n_anchratios,
			best_dx_for_bbox):
		"""We ensure that every bbox has at least one positive RPN region."""

		for idx in range(num_anchors_for_bbox.shape[0]):
			if num_anchors_for_bbox[idx] == 0:
				# no box with an IoU greater than zero ...
				if best_anchor_for_bbox[idx, 0] == -1:
					continue
				y_is_box_valid[
					best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1],
					best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3]
				] = 1
				y_rpn_overlap[
					best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1],
					best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3]
				] = 1
				tmp1 = best_anchor_for_bbox[idx,2]
				tmp2 = best_anchor_for_bbox[idx,3]
				start = 4 * (tmp1 + n_anchratios * tmp2)
				y_rpn_regr[
					best_anchor_for_bbox[idx,0],
					best_anchor_for_bbox[idx,1],
					start:start+4
				] = best_dx_for_bbox[idx, :]

		y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
		y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

		y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
		y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

		y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
		y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

		return y_rpn_overlap, y_is_box_valid, y_rpn_regr

	@staticmethod
	def get_anchor_gt(all_data, class_count, config, img_len_calc_func, mode='train'):
		"""Get anchors of ground truth if the proposal regions are positive."""

		sample_selector = SampleSelector(class_count)
		data_augment_img = DataAugment(config)

		while True:
			if mode == 'train':
				# Randomize all image data
				np.random.shuffle(all_data)

			for img_data in all_data:
				try:
					cond2 = sample_selector.skip_sample_for_balanced_class(img_data)
					if config.balanced_classes and cond2:
						continue

					# read in image, and optionally add augmentation
					if mode == 'train':
						img_data_aug, x_img = data_augment_img.augment(
							img_data,
							augment=True
						)
					else:
						img_data_aug, x_img = data_augment_img.augment(
							img_data,
							augment=False
						)

					(width, height) = (img_data_aug['width'], img_data_aug['height'])
					(rows, cols, _) = x_img.shape
					assert cols == width
					assert rows == height
					# get image dimensions for resizing
					(new_width, new_height) = ImageTools.get_new_img_size(
						width,
						height,
						config.im_size
					)
					# resize the image so that smalles side is length = 600px
					x_img = cv2.resize(
						x_img,
						(new_width, new_height),
						interpolation=cv2.INTER_CUBIC
					)

					try:
						y_rpn_cls, y_rpn_regr = Utils.calc_rpn(
							config,
							img_data_aug,
							width, height,
							new_width, new_height,
							img_len_calc_func
						)
					except Exception as e:
						print("L2 get_anchor_gt", e)
						continue

					# Zero-center by mean pixel, and preprocess image
					x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
					x_img = x_img.astype(np.float32)
					x_img[:, :, 0] -= config.img_channel_mean[0]
					x_img[:, :, 1] -= config.img_channel_mean[1]
					x_img[:, :, 2] -= config.img_channel_mean[2]
					x_img /= config.img_scaling_factor

					x_img = np.transpose(x_img, (2, 0, 1))
					x_img = np.expand_dims(x_img, axis=0)
					y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= config.std_scaling

					# Format for TensorFlow backend
					x_img = np.transpose(x_img, (0, 2, 3, 1))
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

					aux = np.copy(x_img)
					yield aux, [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

				except Exception as e:
					print("L1 get_anchor_gt", e)
					continue
