import numpy as np
import pdb
import math
import copy

from . data_generator import Metrics
from . utilities.image_tools import ImageTools


class ROIHelpers(object):
	"""Assists in the calculation of regions of interest (ROIs)."""

	def __init__(self, config, overlap_thresh=0.9, max_boxes=300):
		super(ROIHelpers, self).__init__()
		self.config = config
		self.overlap_thresh = overlap_thresh
		self.max_boxes = max_boxes

	def set_overlap_thresh(self, new_overlap_thresh):
		self.overlap_thresh = new_overlap_thresh

	def calc_iou(self, R, data, class_mapping):
		"""Calc the best IoUs considering all classes."""

		bboxes = data['bboxes']
		(width, height) = (data['width'], data['height'])
		# get image dimensions for resizing
		(new_width, new_height) = ImageTools.get_new_img_size(width, height, self.config.im_size)

		gta = np.zeros((len(bboxes), 4))

		# get the GT box coordinates, and resize to account for image resizing
		for bbox_num, bbox in enumerate(bboxes):
			rpn_stride = self.config.rpn_stride
			gta[bbox_num, 0] = int(
				round(bbox['x1'] * (new_width / float(width)) / rpn_stride)
			)
			gta[bbox_num, 1] = int(
				round(bbox['x2'] * (new_width / float(width)) / rpn_stride)
			)
			gta[bbox_num, 2] = int(
				round(bbox['y1'] * (new_height / float(height)) / rpn_stride)
			)
			gta[bbox_num, 3] = int(
				round(bbox['y2'] * (new_height / float(height)) / rpn_stride)
			)

		x_roi = []
		y_class_num = []
		y_class_regr_coords = []
		y_class_regr_label = []
		# for debugging only
		IoUs = []

		for ix in range(R.shape[0]):
			(x1, y1, x2, y2) = R[ix, :]
			x1 = int(round(x1))
			y1 = int(round(y1))
			x2 = int(round(x2))
			y2 = int(round(y2))
			# Get the best bounding box, i.e. best IoU
			best_iou = 0.0
			best_bbox = -1
			for bbox_num in range(len(bboxes)):
				curr_iou = Metrics.iou(
					[
						gta[bbox_num, 0],
						gta[bbox_num, 2],
						gta[bbox_num, 1],
						gta[bbox_num, 3]
					],
					[x1, y1, x2, y2]
				)
				if curr_iou > best_iou:
					best_iou = curr_iou
					best_bbox = bbox_num

			if best_iou < self.config.classifier_min_overlap:
				continue
			else:
				w = x2 - x1
				h = y2 - y1
				x_roi.append([x1, y1, w, h])
				IoUs.append(best_iou)
				min_overlap = self.config.classifier_min_overlap
				max_overlap = self.config.classifier_max_overlap
				if min_overlap <= best_iou < max_overlap:
					# hard negative example, background
					cls_name = 'bg'
				elif max_overlap <= best_iou:
					cls_name = bboxes[best_bbox]['class']
					cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
					cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0
					cx = x1 + w / 2.0
					cy = y1 + h / 2.0
					tx = (cxg - cx) / float(w)
					ty = (cyg - cy) / float(h)
					tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
					th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
				else:
					print('roi = {}'.format(best_iou))
					raise RuntimeError

			class_num = class_mapping[cls_name]
			class_label = len(class_mapping) * [0]
			class_label[class_num] = 1
			y_class_num.append(copy.deepcopy(class_label))
			coords = [0] * 4 * (len(class_mapping) - 1)
			labels = [0] * 4 * (len(class_mapping) - 1)
			if cls_name != 'bg':
				label_pos = 4 * class_num
				sx, sy, sw, sh = self.config.classifier_regr_std
				coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
				labels[label_pos:4+label_pos] = [1, 1, 1, 1]
				y_class_regr_coords.append(copy.deepcopy(coords))
				y_class_regr_label.append(copy.deepcopy(labels))
			else:
				y_class_regr_coords.append(copy.deepcopy(coords))
				y_class_regr_label.append(copy.deepcopy(labels))

		if len(x_roi) == 0:
			return None, None, None, None

		X = np.array(x_roi)
		Y1 = np.array(y_class_num)
		Y2 = np.concatenate(
			[np.array(y_class_regr_label), np.array(y_class_regr_coords)],
			axis=1
		)
		x1_ans = np.expand_dims(X, axis=0)
		return x1_ans, np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs

	@staticmethod
	def apply_regr(x, y, w, h, tx, ty, tw, th):
		"""Apply regression, calc rectangle tighter."""

		try:
			cx = x + w/2.
			cy = y + h/2.
			cx1 = tx * w + cx
			cy1 = ty * h + cy
			w1 = math.exp(tw) * w
			h1 = math.exp(th) * h
			x1 = cx1 - w1/2.
			y1 = cy1 - h1/2.
			x1 = int(round(x1))
			y1 = int(round(y1))
			w1 = int(round(w1))
			h1 = int(round(h1))

			return x1, y1, w1, h1

		except ValueError:
			return x, y, w, h
		except OverflowError:
			return x, y, w, h
		except Exception as e:
			print(e)
			return x, y, w, h

	@staticmethod
	def apply_regr_np(X, T):

		"""Apply regression, calc rectangle tighter, passing numpy vectors."""
		try:
			x = X[0, :, :]
			y = X[1, :, :]
			w = X[2, :, :]
			h = X[3, :, :]

			tx = T[0, :, :]
			ty = T[1, :, :]
			tw = T[2, :, :]
			th = T[3, :, :]

			cx = x + w/2.
			cy = y + h/2.
			cx1 = tx * w + cx
			cy1 = ty * h + cy

			w1 = np.exp(tw.astype(np.float64)) * w
			h1 = np.exp(th.astype(np.float64)) * h
			x1 = cx1 - w1/2.
			y1 = cy1 - h1/2.

			x1 = np.round(x1)
			y1 = np.round(y1)
			w1 = np.round(w1)
			h1 = np.round(h1)
			return np.stack([x1, y1, w1, h1])
		except Exception as e:
			print(e)
			return X

	def apply_non_max_suppression_fast(self, boxes, probs):
		"""Select the bounding boxes most likely, deleting the rest."""

		# If there are no boxes, return an empty list
		if len(boxes) == 0:
			return []
		# grab the coordinates of the bounding boxes
		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]

		np.testing.assert_array_less(x1, x2)
		np.testing.assert_array_less(y1, y2)

		# if the bounding boxes integers, convert them to floats.
		# this is important since we'll be doing a bunch of divisions
		if boxes.dtype.kind == "i":
			boxes = boxes.astype("float")

		# initialize the list of picked indexes
		pick = []
		# calculate the areas
		area = (x2-x1) * (y2-y1)
		# sort the bounding boxes
		idxs = np.argsort(probs)

		# keep looping while some indexes still remain in the indexes list
		while len(idxs) > 0:
			""" Grab the last index in the indexes list and add the index value
			to the list of picked indexes.
			"""
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)

			# find the intersection
			xx1_int = np.maximum(x1[i], x1[idxs[:last]])
			yy1_int = np.maximum(y1[i], y1[idxs[:last]])
			xx2_int = np.minimum(x2[i], x2[idxs[:last]])
			yy2_int = np.minimum(y2[i], y2[idxs[:last]])
			# Width and Height
			ww_int = np.maximum(0, xx2_int - xx1_int)
			hh_int = np.maximum(0, yy2_int - yy1_int)

			area_int = ww_int * hh_int

			# find the union
			area_union = area[i] + area[idxs[:last]] - area_int

			# compute the ratio of overlap
			overlap = area_int / (area_union + 1e-6)

			# delete all indexes from the index list that have
			idxs = np.delete(
				idxs,
				np.concatenate(([last], np.where(overlap > self.overlap_thresh)[0]))
			)

			if len(pick) >= self.max_boxes:
				break

		# Return only the bbxes that were picked using the integer data type
		boxes = boxes[pick].astype("int")
		probs = probs[pick]
		return boxes, probs

	def convert_rpn_to_roi(self, rpn_layer, regr_layer, use_regr=True):
		"""Convert a proposal region into a region of interest."""

		regr_layer = regr_layer / self.config.std_scaling
		anchor_sizes = self.config.anchor_box_scales
		anchor_ratios = self.config.anchor_box_ratios

		assert rpn_layer.shape[0] == 1
		# Recover rows and cols according to TF Backend
		(rows, cols) = rpn_layer.shape[1:3]

		curr_layer = 0
		A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

		for anchor_size in anchor_sizes:
			for anchor_ratio in anchor_ratios:
				anchor_x = (anchor_size * anchor_ratio[0])
				anchor_x /= self.config.rpn_stride
				anchor_y = (anchor_size * anchor_ratio[1])
				anchor_y /= self.config.rpn_stride

				regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
				regr = np.transpose(regr, (2, 0, 1))
				X, Y = np.meshgrid(np.arange(cols),np.arange(rows))

				A[0, :, :, curr_layer] = X - anchor_x/2
				A[1, :, :, curr_layer] = Y - anchor_y/2
				A[2, :, :, curr_layer] = anchor_x
				A[3, :, :, curr_layer] = anchor_y

				if use_regr:
					A[:, :, :, curr_layer] = ROIHelpers.apply_regr_np(
						A[:, :, :, curr_layer],
						regr
					)

				A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
				A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
				A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
				A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

				A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
				A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
				A[2, :, :, curr_layer] = np.minimum(
					cols-1,
					A[2, :, :, curr_layer]
				)
				A[3, :, :, curr_layer] = np.minimum(
					rows-1,
					A[3, :, :, curr_layer]
				)
				curr_layer += 1

		all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
		all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

		x1 = all_boxes[:, 0]
		y1 = all_boxes[:, 1]
		x2 = all_boxes[:, 2]
		y2 = all_boxes[:, 3]

		idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

		all_boxes = np.delete(all_boxes, idxs, 0)
		all_probs = np.delete(all_probs, idxs, 0)

		result = self.apply_non_max_suppression_fast(all_boxes, all_probs)[0]
		return result
