# -*- coding: utf-8 -*-

from __future__ import division
import os
import sys
import time
import pickle
import logging
from optparse import OptionParser

import cv2
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt

from frcnn.cnn import CNN
from frcnn.roi_helpers import ROIHelpers
from frcnn.data_generator import Metrics
from frcnn.utilities.config import Config
from frcnn.utilities.parser import Parser
from frcnn.utilities.image_tools import ImageTools
from frcnn.utilities.config import Config

class ShapeClassifier(object):
	"""Shape Classifier, detects elements of handwritten flowchart using a pre-
	trained model and using Faster R-CNN like architecture.
	"""

	def __init__(self, results_path, bbox_threshold=0.5):
		super(ShapeClassifier, self).__init__()
		self.results_path = results_path
		self.config = None
		self.__load_config(results_path)
		self.class_mapping = self.config.class_mapping
		self.bbox_threshold = bbox_threshold
		# Invert dictionary of classes
		self.class_mapping = {v: k for k, v in self.class_mapping.items()}
		# Assign a color for each class
		self.colors_class = {
			self.class_mapping[v]:
				np.random.randint(0, 255, 3) for v in self.class_mapping
		}
		# Build Faster R-CNN
		self.__build_frcnn()

	def predict(self, image, image_name, save_image=True):
		"""Perform object detection, in this case elements of flowchart."""

		st = time.time() # start time
		# Format input image
		X, ratio = ImageTools.get_format_img_size(image, self.config)
		X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2, F] = self.model_rpn.predict(X)

		# Instance a ROI Heper
		roi_helper = ROIHelpers(
			self.config,
			overlap_thresh=0.5,
		)
		R = roi_helper.convert_rpn_to_roi(Y1, Y2)
		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]
		# Apply the spatial pyramid pooling to the proposed regions
		bboxes, probs = self.__apply_spatial_pyramid_pooling(R, F)

		img, new_boxes, new_probs = self.__generate_final_image(
			bboxes,
			probs,
			image,
			roi_helper,
			ratio
		)

		print('Elapsed time = {}'.format(time.time() - st))
		if(save_image):
			filepath = self.results_path + "/testing/"
			cv2.imwrite(filepath + image_name, img)
			print("Image {}, save in {}".format(image_name, filepath))

	def __load_config(self, results_path):
		"""Open .pickle file that contains configuration params of F R-CNN."""

		config_path = results_path + "/config.pickle"
		try:
			with open(config_path, 'rb') as f_in:
				self.config = pickle.load(f_in)
				print("Config loaded successful!!")
		except Exception as e:
			print("Could not load configuration file, ¡check results path!")

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

		model_path = self.config.weights_output_path
		try:
			print('Loading weights from {}'.format(model_path))
			self.model_rpn.load_weights(model_path, by_name=True)
			self.model_classifier.load_weights(model_path, by_name=True)
		except Exception as e:
			print('Exception: {}'.format(e))
			print("Couldn't load pretrained model weights!")

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
		new_boxes = []
		new_probs = []

		for key in bboxes:
			bbox = np.array(bboxes[key])
			# apply non max suppression algorithm
			roi_helper.set_overlap_thresh(0.7)
			new_boxes, new_probs = roi_helper.apply_non_max_suppression_fast(
				bbox,
				np.array(probs[key])
			)

			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]

				real_coords = ImageTools.get_real_coordinates(ratio, x1, y1, x2, y2)
				real_x1, real_y1, real_x2, real_y2 = real_coords
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

				textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
				all_dets.append((key, 100 * new_probs[jk]))

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
		print(all_dets)
		return img, new_boxes, new_probs

if __name__ == '__main__':
	classifier = ShapeClassifier("training_results/7")

	base_path = "/home/david/Escritorio/images_test_flowchart-3b/"

	for i in range(14):
		img_path = base_path + str(i) + ".jpg"
		img = cv2.imread(img_path)
		classifier.predict(img, str(i) + ".jpg")
