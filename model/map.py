# -*- coding: utf-8 -*-

"""MAP (mean Average Precision) is a popular metric in measuring the
accuracy of object detectors like Faster R-CNN, SSD, etc. Average precision
computes the average precision value for recall value over 0 to 1.
"""
import os
import cv2
import sys
import time
import pickle
import random
from optparse import OptionParser

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from sklearn.metrics import average_precision_score

from frcnn.cnn import CNN
from frcnn.roi_helpers import ROIHelpers
from frcnn.data_generator import Metrics
from frcnn.utilities.config import Config
from frcnn.utilities.parser import Parser
from frcnn.utilities.image_tools import ImageTools

class MAP(object):
	"""Evaluate with mAP a model pre-trained."""

	def __init__(self, annotate_path, config_path):
		super(MAP, self).__init__()
		self.annotate_path = annotate_path
		self.config = None
		# Load config file
		self.__load_config(config_path)
		# Prepare class mapping
		self.class_mapping = self.config.class_mapping
		if 'bg' not in self.class_mapping:
			self.class_mapping['bg'] = len(self.class_mapping)
		self.class_mapping = {v: k for k, v in self.class_mapping.items()}
		# Models
		self.model_rpn = None
		self.model_classifier_only = None
		self.model_classifier = None
		# Build Faster R-CNN
		self.__build_frcnn()
		# Load data from annotation file (txt)
		self.test_images = []
		self.__load_data()

	def __load_config(self, config_path):
		with open(config_path, 'rb') as f_in:
			self.config = pickle.load(f_in)

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
		cnn = CNN(num_anchors, (roi_input, self.config.num_rois), len(self.class_mapping))
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

	def __load_data(self):
		parser = Parser(
			dataset_path="/home/david/Escritorio/flowchart-3b(splitter)",
			annotate_path=self.annotate_path,
		)
		# Recover image paths
		all_imgs, _, _ = parser.get_data(generate_annotate=False)
		self.test_images = [s for s in all_imgs if s['imageset'] == 'test']
		random.shuffle(self.test_images)

	def measure_map(self):
		"""Measure AP for each test/validation images class and finally the
		mAP.
		"""

		GT = {}
		Predicted = {} # confidence score for each
		cont = 0
		# Iterate above all images
		for idx, img_data in enumerate(self.test_images):
			print('{}/{}'.format(idx,len(self.test_images)))
			st = time.time()

			filepath = img_data['filepath']
			img = cv2.imread(filepath)
			# The resized image and the factors for each dimension are obtained-
			X, fx, fy = ImageTools.format_img(img, self.config)
			X = np.transpose(X, (0, 2, 3, 1))

			# get the feature maps and output from the RPN
			[Y1, Y2, F] = self.model_rpn.predict(X)
			# Instance a ROI Heper
			roi_helper = ROIHelpers(
				self.config,
				overlap_thresh=0.7,
			)
			R = roi_helper.convert_rpn_to_roi(Y1, Y2)

			# convert from (x1,y1,x2,y2) to (x,y,w,h)
			R[:, 2] -= R[:, 0]
			R[:, 3] -= R[:, 1]

			bboxes, probs = self.__apply_spatial_pyramid_pooling(R, F)

			all_dets = self.__apply_non_max_sup_for_each_class(
				bboxes,
				probs,
				roi_helper
			)
			print('Elapsed time = {}'.format(time.time() - st))

			t, p = self.__get_map(all_dets, img_data['bboxes'], (fx, fy))

			for key in t.keys():
				if key not in GT:
					GT[key] = []
					Predicted[key] = []
				GT[key].extend(t[key])
				Predicted[key].extend(p[key])
			all_aps = []
			for key in GT.keys():
				ap = average_precision_score(GT[key], Predicted[key])
				print('{} AP: {}'.format(key, ap))
				all_aps.append(ap)
			print('mAP = {}'.format(np.mean(np.array(all_aps))))
			cont += 1
			print(GT)
			print(Predicted)
		print("Number of test images: ", cont)

	def __apply_spatial_pyramid_pooling(self, roi, F):
		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}
		num_rois = self.config.num_rois

		for jk in range(roi.shape[0] // num_rois + 1):
			ROIs = np.expand_dims(
				roi[num_rois * jk:num_rois * (jk + 1), :],
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

				if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
					tx /= self.config.classifier_regr_std[0]
					ty /= self.config.classifier_regr_std[1]
					tw /= self.config.classifier_regr_std[2]
					th /= self.config.classifier_regr_std[3]
					x, y, w, h = ROIHelpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append(
					[16 * x, 16 * y, 16 * (x + w), 16 * (y + h)]
				)
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		return bboxes, probs

	def __apply_non_max_sup_for_each_class(self, bboxes, probs, roi_helper):
		all_dets = []
		# Apply non max suppression for each class
		for key in bboxes:
			bbox = np.array(bboxes[key])
			roi_helper.set_overlap_thresh(0.1)
			new_boxes, new_probs = roi_helper.apply_non_max_suppression_fast(
				bbox,
				np.array(probs[key])
			)
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk, :]
				det = {
					'x1': x1,
					'x2': x2,
					'y1': y1,
					'y2': y2,
					'class': key,
					'prob': new_probs[jk]
				}
				all_dets.append(det)

		return all_dets

	def __get_map(self, pred, gt, factors):
		T = {}
		P = {}
		fx, fy = factors

		for bbox in gt:
			bbox['bbox_matched'] = False

		pred_probs = np.array([s['prob'] for s in pred])
		box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

		for box_idx in box_idx_sorted_by_prob:
			pred_box = pred[box_idx]
			pred_class = pred_box['class']
			pred_x1 = pred_box['x1']
			pred_x2 = pred_box['x2']
			pred_y1 = pred_box['y1']
			pred_y2 = pred_box['y2']
			pred_prob = pred_box['prob']
			if pred_class not in P:
				P[pred_class] = []
				T[pred_class] = []
			P[pred_class].append(pred_prob)
			found_match = False

			for gt_box in gt:
				gt_class = gt_box['class']
				gt_x1 = gt_box['x1'] / fx
				gt_x2 = gt_box['x2'] / fx
				gt_y1 = gt_box['y1'] / fy
				gt_y2 = gt_box['y2'] / fy
				gt_seen = gt_box['bbox_matched']
				if gt_class != pred_class:
					continue
				if gt_seen:
					continue
				iou = Metrics.iou(
					(pred_x1, pred_y1, pred_x2, pred_y2),
					(gt_x1, gt_y1, gt_x2, gt_y2)
				)
				if iou >= 0.5:
					found_match = True
					gt_box['bbox_matched'] = True
					break
				else:
					continue

			T[pred_class].append(int(found_match))

		for gt_box in gt:
			if not gt_box['bbox_matched']:
				if gt_box['class'] not in P:
					P[gt_box['class']] = []
					T[gt_box['class']] = []

				T[gt_box['class']].append(1)
				P[gt_box['class']].append(0)

		return T, P

if __name__ == '__main__':
	results_path = "training_results/5"
	annotate_path = results_path + "/annotate2.txt"
	config_path = results_path + "/config.pickle"
	map = MAP(annotate_path=annotate_path, config_path=config_path)

	map.measure_map()
