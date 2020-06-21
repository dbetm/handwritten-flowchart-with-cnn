# -*- coding: utf-8 -*-

from __future__ import division
import os
import sys
import time
import copy
import pickle
import random
import logging
from optparse import OptionParser

import cv2
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

from . frcnn.cnn import CNN
from . frcnn.roi_helpers import ROIHelpers
from . frcnn.data_generator import Metrics
from . frcnn.utilities.config import Config
from . frcnn.utilities.parser import Parser
from . frcnn.utilities.image_tools import ImageTools
sys.path.append("model")
import frcnn
sys.path.append('..')
from node import Node

class ShapeClassifier(object):
	"""Shape Classifier, detects elements of handwritten flowchart using a pre-
	trained model and using Faster R-CNN like architecture.
	"""

	def __init__(
		self,
		results_path,
		bbox_threshold=0.5,
		overlap_thresh_1=0.5,
		overlap_thresh_2=0.3,
		use_gpu=False,
		num_rois=0
		):
		super(ShapeClassifier, self).__init__()
		self.results_path = results_path
		self.config = None
		self.__load_config(results_path)
		self.class_mapping = self.config.class_mapping
		# Override num_rois if it was specified
		if(num_rois > 0):
			self.config.num_rois = num_rois
		# Thresholds
		self.bbox_threshold = bbox_threshold
		self.overlap_thresh_1 = overlap_thresh_1
		self.overlap_thresh_2 = overlap_thresh_2
		# Invert dictionary of classes
		self.class_mapping = {v: k for k, v in self.class_mapping.items()}
		# Assign a color for each class
		self.colors_class = {
			self.class_mapping[v]:
				np.random.randint(0, 255, 3) for v in self.class_mapping
		}
		if(use_gpu):
			self.__setup()
		# Build Faster R-CNN
		self.__build_frcnn()
		self.removable_threshold = 49.0
		# saving tmp rectangles, path to save
		self.RECTANGLES_PATH = "../Images/tmp/"

	def __setup(self):
		config_gpu = tf.compat.v1.ConfigProto()
		# dynamically grow the memory used on the GPU
		config_gpu.gpu_options.allow_growth = True
		# to log device placement (on which device the operation ran)
		config_gpu.log_device_placement = True
		sess = tf.compat.v1.Session(config=config_gpu)

	def predict_and_save(self, image, image_name, folder_name):
		"""Perform object detection, in this case elements of flowchart and
		draw bounding boxes and image and save the same."""

		st = time.time() # start time
		# Format input image
		X, ratio = ImageTools.get_format_img_size(image, self.config)
		X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2, F] = self.model_rpn.predict(X)

		# Instance a ROI Heper
		roi_helper = ROIHelpers(
			self.config,
			overlap_thresh=self.overlap_thresh_1
		)
		R = roi_helper.convert_rpn_to_roi(Y1, Y2)
		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]
		# Apply the spatial pyramid pooling to the proposed regions
		bboxes, probs = self.__apply_spatial_pyramid_pooling(R, F)

		img, _ = self.__generate_final_image(
			bboxes,
			probs,
			image,
			roi_helper,
			ratio
		)

		print('Elapsed time = {}'.format(time.time() - st))
		# Save image
		path = self.results_path + "/" + folder_name
		if(os.path.isdir(path) == False):
			os.mkdir(path)

		cv2.imwrite(path + "/" + image_name, img)
		print("Image {}, save in {}".format(image_name, path))

	def predict(self, image, display_image):
		"""Object detection for flowchart and generate nodes for shapes
		and connectors."""

		# Format input image
		X, ratio = ImageTools.get_format_img_size(image, self.config)
		X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2, F] = self.model_rpn.predict(X)
		# Instance a ROI Heper
		roi_helper = ROIHelpers(
			self.config,
			overlap_thresh=self.overlap_thresh_1
		)
		R = roi_helper.convert_rpn_to_roi(Y1, Y2)
		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]
		# Apply the spatial pyramid pooling to the proposed regions
		bboxes, probs = self.__apply_spatial_pyramid_pooling(R, F)

		img, all_dets = self.__generate_final_image(
			bboxes,
			probs,
			image,
			roi_helper,
			ratio
		)
		if(display_image):
			cv2.imshow('test', cv2.resize(img,(0,0), fx=0.4, fy=0.4))
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return self.generate_nodes(all_dets)

	def __load_config(self, results_path):
		"""Open .pickle file that contains configuration params of Faster
		R-CNN.
		"""

		config_path = results_path + "/config.pickle"
		try:
			objects = []
			with (open(config_path, 'rb')) as f_in:
				self.config = pickle.load(f_in)
			print("Config loaded successful!!")
		except Exception as e:
			print("Could not load configuration file, check results path!")
			exit()

	def __get_real_coordinates(ratio, x1, y1, x2, y2):
		"""Method to transform the coordinates of the bounding box to its
		original size.
		"""

		real_x1 = int(round(x1 // ratio))
		real_y1 = int(round(y1 // ratio))
		real_x2 = int(round(x2 // ratio))
		real_y2 = int(round(y2 // ratio))

		return (real_x1, real_y1, real_x2 ,real_y2)

	def __build_frcnn(self):
		"""Build models of the Faster R-CNN."""

		input_shape_img = (None, None, 3)
		input_shape_features = (None, None, 512)

		img_input = Input(shape=input_shape_img)
		roi_input = Input(shape=(self.config.num_rois, 4))
		feature_map_input = Input(shape=input_shape_features)

		num_anchors = len(self.config.anchor_box_scales)
		num_anchors *= len(self.config.anchor_box_ratios)
		# Define the base (layers) network (VGG16).
		cnn = CNN(
			num_anchors,
			(roi_input, self.config.num_rois),
			len(self.class_mapping)
		)
		shared_layers = cnn.build_nn_base(img_input)
		# Define the RPN, built on the base layers
		rpn_layers = cnn.create_rpn(shared_layers)
		# Create classifier
		classifier = cnn.build_classifier(
			feature_map_input,
			len(self.class_mapping)
		)
		# Create models
		self.model_rpn = Model(img_input, rpn_layers)
		self.model_classifier_only = Model(
			[feature_map_input, roi_input],
			classifier
		)
		self.model_classifier = Model([feature_map_input, roi_input], classifier)
		# Load weights from pre-trained model
		self.__load_weights()
		# Compile models
		self.__compile_models()

	def __load_weights(self):
		"""Load weights for pre-trained models."""
		# uncomment for integration with handler
		#model_path = "model/"+self.config.weights_output_path
		# comment for integration
		model_path = "model/"+self.config.weights_output_path

		try:
			print('Loading weights from {}'.format(model_path))
			self.model_rpn.load_weights(model_path, by_name=True)
			self.model_classifier.load_weights(model_path, by_name=True)
		except Exception as e:
			print('Exception: {}'.format(e))
			print("Couldn't load pretrained model weights!")
			exit()

	def __compile_models(self):
		"""Compile the models."""

		self.model_rpn.compile(optimizer='sgd', loss='mse')
		self.model_classifier.compile(optimizer='sgd', loss='mse')

	def __apply_spatial_pyramid_pooling(self, roi, F):
		"""Pools the last feature map in a way that will generate
		fixed length vectors for the fully connected layers. With this
		Spatial Pyramid pooling (see Fig.8) there is no need to warp or
		crop the inputted images."""

		bboxes = {}
		probs = {}
		bbox_threshold = self.bbox_threshold
		num_rois = self.config.num_rois

		for jk in range(roi.shape[0] // num_rois + 1):
			ROIs = np.expand_dims(
				roi[num_rois * jk:num_rois * (jk+1), :],
				axis=0
			)
			if ROIs.shape[1] == 0:
				break

			if jk == roi.shape[0] // num_rois:
				# padding R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0], num_rois, curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

			for ii in range(P_cls.shape[1]):
				cond1 = np.max(P_cls[0, ii, :]) < bbox_threshold
				if cond1 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num+1)]
					tx /= self.config.classifier_regr_std[0]
					ty /= self.config.classifier_regr_std[1]
					tw /= self.config.classifier_regr_std[2]
					th /= self.config.classifier_regr_std[3]
					x, y, w, h = ROIHelpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass

				stride = self.config.rpn_stride
				bboxes[cls_name].append(
					[stride * x, stride * y, stride * (x+w), stride * (y+h)]
				)
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		return bboxes, probs

	def __generate_final_image(self, bboxes, probs, img, roi_helper, ratio):
		"""Add rectangles of bounding boxes of task detection in
		original image, add caption and probability of classification.
		"""

		all_dets = []
		original_img = copy.copy(img)
		i = -1
		for key in bboxes:

			bbox = np.array(bboxes[key])
			# apply non max suppression algorithm
			roi_helper.set_overlap_thresh(self.overlap_thresh_2)
			new_boxes, new_probs = roi_helper.apply_non_max_suppression_fast(
				bbox,
				np.array(probs[key])
			)

			for jk in range(new_boxes.shape[0]):
				i += 1
				(x1, y1, x2, y2) = new_boxes[jk,:]

				real_coords = ImageTools.get_real_coordinates(ratio, x1, y1, x2, y2)
				real_x1, real_y1, real_x2, real_y2 = real_coords
				all_dets.append((key, 100 * new_probs[jk], real_coords))

		all_dets = self.__fix_detection(all_dets)
		img = self.__draw_rectangles(all_dets, img)

		return img, all_dets

	def __draw_rectangles(self, all_dets, img):

		for key,prob,coords in all_dets:
			real_x1, real_y1, real_x2, real_y2 = coords

			# Rectangle for shape or connector
			cv2.rectangle(
				img,
				(real_x1, real_y1),
				(real_x2, real_y2),
				(
					int(self.colors_class[key][0]),
					int(self.colors_class[key][1]),
					int(self.colors_class[key][2])
				),
				4
			)

			textLabel = '{}: {}'.format(key, int(prob))
			(retval, baseLine) = cv2.getTextSize(
				textLabel,
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				1
			)
			textOrg = (real_x1, real_y1)
			# Rectangle for text
			cv2.rectangle(
				img,
				(textOrg[0] - 5, textOrg[1] + baseLine - 5),
				(textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
				(22, 166, 184),
				2
			)
			# Fill text rectangle
			cv2.rectangle(
				img,
				(textOrg[0] - 5, textOrg[1] + baseLine - 5),
				(textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
				(255, 255, 255),
				-1
			)
			# Put class text
			cv2.putText(
				img,
				textLabel,
				textOrg,
				cv2.FONT_HERSHEY_DUPLEX,
				1,
				(0, 0, 0),
				1
			)

		return img

	def __fix_detection(self, all_dets):

		index_to_del = []
		x = 0
		for key,prob,coords in all_dets:
			if(x in index_to_del):
				x += 1
				continue

			y = 0
			x1, y1, x2, y2 = coords

			for key_2,prob_2,coords_2 in all_dets:
				if(x == y or y in index_to_del):
					y += 1
					continue

				flag = False
				offset = False
				no_offset = False
				self.removable_threshold = 30.0

				htal = key_2 == 'arrow_line_left' or key_2 == 'arrow_line_right'
				vtal = key_2 == 'arrow_line_up' or key_2 == 'arrow_line_down'

				# Calculate coordinate offset in doble detection
				if(key == key_2):
					width = x2 - x1
					height = y2 - y1
					if(key == 'start_end'):
						coords_pos = (x1 + width*0.2, y1, x2 + width*0.2, y2)
						coords_neg = (x1 - width*0.2, y1, x2 - width*0.2, y2)
					elif(key == 'print' or key == 'scan' or key == 'process'):
						coords_pos = (x1 + width*0.05, y1, x2 + width*0.05, y2)
						coords_neg = (x1 - width*0.05, y1, x2 - width*0.05, y2)
					elif(key == 'arrow_line_right' or key == 'arrow_line_left'):
						coords_pos = (x1, y1 + height*0.2, x2, y2 + height*0.2)
						coords_neg = (x1, y1 - height*0.2, x2, y2 - height*0.2)
					elif(key == 'arrow_line_up' or key == 'arrow_line_down'):
						coords_pos = (x1 + width*0.2, y1, x2 + width*0.2, y2)
						coords_neg = (x1 - width*0.2, y1, x2 - width*0.2, y2)

				if(key == 'print'):
					if(key_2 == 'process' or htal or key_2 == 'start_end'):
						no_offset = True
					elif(key_2 == 'print'):
						offset = True
				elif(key == 'start_end'):
					if(key_2 == 'process' or htal):
						no_offset = True
					elif(key_2 == 'start_end'):
						offset = True
				elif(key == 'scan'):
					if(key_2 == 'process'):
						no_offset = True
					elif(key_2 == 'scan'):
						offset = True
				elif(key == 'process'):
					if(key_2 == 'scan' or htal):
						no_offset = True
					elif(key_2 == 'process'):
						offset = True
				elif(key == 'arrow_line_right'):
					if(key_2 == 'arrow_line_left'):
						no_offset = True
					elif(key_2 == 'arrow_line_right'):
						offset = True
						self.removable_threshold = 15.0
				elif(key == 'arrow_line_left'):
					if(key_2 == 'arrow_line_right'):
						no_offset = True
					elif(key_2 == 'arrow_line_left'):
						offset = True
						self.removable_threshold = 15.0
				elif(key == 'arrow_line_up'):
					if(key_2 == 'arrow_line_down'):
						no_offset = True
					elif(key_2 == 'arrow_line_up'):
						offset = True
						self.removable_threshold = 12.0
				elif(key == 'arrow_line_down'):
					if(key_2 == 'arrow_line_up'):
						no_offset = True
					elif(key_2 == 'arrow_line_down'):
						offset = True
						self.removable_threshold = 12.0
				if(no_offset):
					if(self.__is_bbox_removable(coords, coords_2)):
						flag = True
				elif(offset):
					if(self.__is_bbox_removable(coords_pos, coords_2)):
						flag = True
					elif(self.__is_bbox_removable(coords_neg, coords_2)):
						flag = True

				if(flag):
					if(prob < prob_2):
						index_to_del.append(x)
						break
					else:
						index_to_del.append(y)
				y += 1
			x += 1

		_all_dets = []
		x = 0
		for key,prob,coords in all_dets:
			if(x in index_to_del):
				x += 1
				continue
			_all_dets.append((key, prob, coords))
			x += 1

		return _all_dets

	def __is_bbox_removable(self, coords, coords_2):
		"""Check if exists an area overlap greater than a threshold
		for remove an bounding box."""

		inter_area = Metrics.intersection(coords, coords_2)
		x1, y1, x2, y2 = coords_2
		area_2 = float((x2 - x1) * (y2 - y1))
		percent_area = float((inter_area * 100)) / float(area_2)

		return percent_area > self.removable_threshold

	def generate_nodes(self, dets):
		"""Generate nodes with detections."""

		nodes = []
		for det in dets:
			x1, y1, x2, y2 = det[2]
			cord = [x1,x2,y1,y2]
			node = Node(coordinate=cord, class_shape=det[0])
			nodes.append(node)

		return nodes

	def __is_rectangle_arrow(self, class_shape):
		ans = True
		str = class_shape.split('_')
		if(len(str) < 3):
			ans = False
		elif not('rectangle' in str):
			ans = False

		return ans


if __name__ == '__main__':
	# folder_number = input("Type num folder of training results: ")
	# params_test = int(input("Type params_test (1,2,3): "))
	# test = int(input("Type number of set (1,2,3): "))
	#
	# if(params_test == 1):
	# 	overlap_thresh_1 = 0.9
	# 	overlap_thresh_2 = 0.2
	# 	bbox_threshold = 0.51
	# 	folder_name = "test_1"
	# elif(params_test == 2):
	# 	overlap_thresh_1 = 0.7
	# 	overlap_thresh_2 = 0.05
	# 	bbox_threshold = 0.45
	# 	folder_name = "test_2"
	# else:
	# 	overlap_thresh_1 = 0.9
	# 	overlap_thresh_2 = 0.1
	# 	bbox_threshold = 0.6
	# 	folder_name = "test_3"

	"""overlap_thresh_1 = 0.7
	overlap_thresh_2 = 0.05
	bbox_threshold = 0.45

	img_path = "/home/david/Escritorio/set12/11.jpg"

	# The best model is in folder 9 (training results)
	classifier = ShapeClassifier(
		results_path="training_results/9",
		use_gpu=False,
		overlap_thresh_1=overlap_thresh_1,
		overlap_thresh_2=overlap_thresh_2,
		bbox_threshold=bbox_threshold,
		num_rois=32
	)
	image = cv2.imread(img_path)
	nodes = classifier.predict(image, True)
	print(*enumerate(nodes))"""


	# test_path = "/home/david/Escritorio/set" + str(test) + "/"
	# folder_name += "_" + str(test)

	# img_path = test_path + "rect.jpg"
	# img = cv2.imread(img_path)
	# nodes = classifier.predict(img, display_image=True)
	# print(*nodes)

	# for i in range(len(samples)):
	# 	img_path = test_path + samples[i]
	# 	img = cv2.imread(img_path)
	# 	classifier.predict(img, display_image=False)

	# classifier = ShapeClassifier(
	# 	results_path="training_results/" + folder_number,
	# 	use_gpu=False,
	# 	overlap_thresh_1=overlap_thresh_1,
	# 	overlap_thresh_2=overlap_thresh_2,
	# 	bbox_threshold=bbox_threshold,
	# 	num_rois=32
	# )
	#
	# limit = 56
	# for i in range(1, limit+1):
	# 	img_path = test_path + str(i) + ".jpg"
	# 	img = cv2.imread(img_path)
	# 	# cv2.imshow('test', img)
	# 	# cv2.waitKey(0)
	# 	# cv2.destroyAllWindows()
	# 	classifier.predict_and_save(img, str(i) + ".jpg", folder_name)
