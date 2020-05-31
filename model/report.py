# -*- coding: utf-8 -*-

import os
import cv2
import sys
import time
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, confusion_matrix

from frcnn.cnn import CNN
from frcnn.roi_helpers import ROIHelpers
from frcnn.data_generator import Metrics
from frcnn.utilities.config import Config
from frcnn.utilities.parser import Parser
from frcnn.utilities.image_tools import ImageTools


class Report(object):
	"""Report generate confusion matrix, calculate mAP and estimate precision
	and recall for each class."""

	def __init__(self, results_path, dataset_path, generate_annotate=False, use_gpu=False):
		super(Report, self).__init__()
		self.results_path = results_path
		self.annotate_path = results_path + "annotate.txt"
		self.config = None
		# Load config file
		self.__load_config()
		self.IOU_THRESHOLD = self.config.classifier_max_overlap
		# Get class mapping from configuration
		self.class_mapping = {v: k for k, v in self.config.class_mapping.items()}
		# Init confusion matrix
		dim_matriz = (len(self.class_mapping), len(self.class_mapping))
		self.cfn_matrix = np.zeros(shape=dim_matriz)
		if(use_gpu):
			self.__setup()
		# Models
		self.model_rpn = None
		self.model_classifier_only = None
		self.model_classifier = None
		# Build Faster R-CNN
		self.__build_frcnn()
		# Load data from annotation file (txt)
		self.test_images = []
		self.__load_data(dataset_path, generate_annotate)

	def __setup(self):
		config_gpu = tf.compat.v1.ConfigProto()
		# dynamically grow the memory used on the GPU
		config_gpu.gpu_options.allow_growth = True
		# to log device placement (on which device the operation ran)
		config_gpu.log_device_placement = True
		sess = tf.compat.v1.Session(config=config_gpu)

	def __load_config(self):
		if not(os.path.isdir(self.results_path)):
			print("Error: Not valid results path!")
			exit()
		config_path = self.results_path + "config.pickle"
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
			exit()

	def __compile_models(self):
		"""Compile the models."""

		self.model_rpn.compile(optimizer='sgd', loss='mse')
		self.model_classifier.compile(optimizer='sgd', loss='mse')

	def __load_data(self, dataset_path, generate_annotate):
		parser = Parser(
			dataset_path=dataset_path,
			annotate_path=self.annotate_path,
		)
		# Recover image paths
		all_imgs, _, _ = parser.get_data(generate_annotate=generate_annotate)
		self.test_images = [s for s in all_imgs if s['imageset'] == 'test']
		random.shuffle(self.test_images)

	def generate(self):
		"""Predict objects and compare with ground-truth for estimate metrics in the
		object detection task.
		"""

		global_time_init = time.time()
		GT = {}
		Predicted = {} # confidence score for each test image
		total_detected = 0
		total_gt = 0
		# Iterate above all images
		for idx, img_data in enumerate(self.test_images):
			print('{}/{}'.format(idx,len(self.test_images)))
			st = time.time()

			filepath = img_data['filepath']
			img = cv2.imread(filepath)

			total_gt += len(img_data['bboxes'])
			# The resized image and the factors for each dimension are obtained-
			X, fx, fy = ImageTools.format_img(img, self.config)
			X = np.transpose(X, (0, 2, 3, 1))

			# get the feature maps and output from the RPN
			[Y1, Y2, F] = self.model_rpn.predict(X)
			# Instance a ROI Heper
			roi_helper = ROIHelpers(
				self.config,
				overlap_thresh=self.IOU_THRESHOLD,
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
			total_detected += len(all_dets)
			print('Elapsed time = {}'.format(time.time() - st))

			t, p = self.__get_map(all_dets, img_data['bboxes'], (fx, fy))
			self.__update_confusion_matrix(all_dets, img_data['bboxes'], (fx, fy))

			for key in t.keys():
				if key not in GT:
					GT[key] = []
					Predicted[key] = []
				GT[key].extend(t[key])
				Predicted[key].extend(p[key])

		msg = "\nAP for each class:\n\n"
		print("."*45 + msg)
		self.mAP_file = open(self.results_path + "mAP.txt", "x")
		self.mAP_file.write(msg)
		# Calculate AP for each class and mAP finally
		# Write results in file mAP.txt
		all_aps = []
		for key in GT.keys():
			ap = average_precision_score(GT[key], Predicted[key])
			print('{} AP: {}'.format(key, ap))
			self.mAP_file.write('{} AP: {}\n'.format(key, ap))
			all_aps.append(ap)

		mAP = np.mean(np.array(all_aps))
		print('mAP = {}'.format(mAP))
		self.mAP_file.write('\nmAP = {}\n'.format(mAP))
		print('Total objects detected = {}'.format(total_detected))
		self.mAP_file.write('Total objects detected = {}\n'.format(total_detected))
		print('Total objects ground-truth = {}'.format(total_gt))
		self.mAP_file.write('Total objects ground-truth = {}\n'.format(total_gt))
		# print(GT)
		# print(Predicted)
		msg = "Number of test images = " + str(len(self.test_images))
		print(msg)
		self.mAP_file.write(msg)
		self.mAP_file.close()
		print('Total elapsed time = {}'.format(time.time() - global_time_init))

		self.__save_classification_report()

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
		all_dets = [] # all detections
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
				if iou >= self.IOU_THRESHOLD:
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

	def __update_confusion_matrix(self, pred, gt, factors):
		fx, fy = factors
		# NOTA: Implementation algorithm based in:
		# https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py
		matches = []
		detection_classes = [d['class'] for d in pred]
		# For each ground-truth object get the IoU with each detected object
		for i in range(len(gt)):
			gt_x1 = gt[i]['x1'] / fx
			gt_x2 = gt[i]['x2'] / fx
			gt_y1 = gt[i]['y1'] / fy
			gt_y2 = gt[i]['y2'] / fy

			for j in range(len(pred)):
				pred_x1 = pred[j]['x1']
				pred_x2 = pred[j]['x2']
				pred_y1 = pred[j]['y1']
				pred_y2 = pred[j]['y2']
				iou = Metrics.iou(
					(pred_x1, pred_y1, pred_x2, pred_y2),
					(gt_x1, gt_y1, gt_x2, gt_y2)
				)
				if iou >= self.IOU_THRESHOLD:
					matches.append([i, j, iou])

		matches = np.array(matches)
		# Prune match list
		if(matches.shape[0] > 0):
			""" Sort list of matches by descending IoU so we can remove
			duplicate detections while keeping the highest IoU entry.
			"""
			matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
			# Remove duplicate detections from the list.
			matches = matches[np.unique(matches[:,1], return_index=True)[1]]
			""" Sort the list again by descending IoU. Removing duplicates
			doesn't preserve our previous sort.
			"""
			matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
			# Remove duplicate ground truths from the list.
			matches = matches[np.unique(matches[:,0], return_index=True)[1]]

		for i in range(len(gt)):
			row = self.config.class_mapping[gt[i]['class']]

			if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
				key = detection_classes[int(matches[matches[:,0] == i, 1][0])]
				col = self.config.class_mapping[key]
				self.cfn_matrix[row][col] += 1
			else:
				self.cfn_matrix[row][self.cfn_matrix.shape[1] - 1] += 1

		for i in range(len(pred)):
			if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
				col = self.config.class_mapping[detection_classes[i]]
				self.cfn_matrix[self.cfn_matrix.shape[0] - 1][col] += 1

	def __save_classification_report(self):
		# y_true = []
		# y_pred = []
		categories = list(self.config.class_mapping.keys())
		# Show confusion matrix (text mode)
		print("."*50 + "\nConfusion matrix")
		print(self.cfn_matrix)
		# Save confusion matrix (txt)
		cnf_matrix_path = self.results_path + "confusion_matrix.txt"
		np.savetxt(cnf_matrix_path, self.cfn_matrix, fmt="%d")
		# Save confusion matrix (png)
		cnf_matrix_path = self.results_path + "confusion_matrix.png"
		fig = plt.figure(figsize=(10,10))
		plt.imshow(self.cfn_matrix, interpolation='nearest')
		plt.colorbar()
		tick_marks = np.arange(len(categories))
		_ = plt.xticks(tick_marks, categories, rotation=90)
		_ = plt.yticks(tick_marks, categories)
		plt.xlabel("Predicted")
		plt.ylabel("Ground-truth")
		plt.tight_layout()
		plt.savefig(cnf_matrix_path, dpi=300)
		# Display confusion matrix image
		plt.show()
		plt.close()
		# Generate results (classification report) and display them
		results = []
		for i in range(len(categories)):
			id = i
			name = categories[id]

			total_target = np.sum(self.cfn_matrix[id,:])
			total_predicted = np.sum(self.cfn_matrix[:,id])

			precision = float(self.cfn_matrix[id, id] / total_predicted)
			recall = float(self.cfn_matrix[id, id] / total_target)

			results.append(
				{
					'category' : name,
					'precision {} IoU'.format(self.IOU_THRESHOLD) : precision,
					'recall {} IoU'.format(self.IOU_THRESHOLD) : recall
				}
			)

		df = pd.DataFrame(results)
		print("."*50 + "\nClassification report")
		print(df)
		classification_report_path = self.results_path + "classification_report.csv"
		df.to_csv(classification_report_path)
		# Generate history loss with history.csv file
		self.generate_graphs_loss_history()

	def generate_graphs_loss_history(self):
		"""Generate two graphs and save like images (.png) for evolution of
		loss metric in training process.
		"""

		file_path = self.results_path + "history.csv"
		if(os.path.isfile(file_path)):
			data = pd.read_csv(file_path)
			losses = np.array(data["loss"])
			epochs = np.array(range(1, len(losses)+1))
			plt.plot(epochs, losses, color='red')
			plt.savefig(self.results_path + "loss_history.png", dpi=300)
			# Generate soft loss history
			soft_losses = []
			local_min = losses[0]
			for x in losses:
				if(x < local_min):
					local_min = x
				soft_losses.append(local_min)

			soft_losses = np.array(soft_losses)
			plt.close()
			plt.plot(epochs, soft_losses, color='green')
			plt.savefig(self.results_path + "soft_loss_history.png", dpi=300)
		else:
			print("history.csv file not found in " + self.results_path)


if __name__ == '__main__':
	results_path = "training_results/3/"
	dataset_path = "/home/david/Escritorio/flowchart_3b_v3"
	report = Report(results_path=results_path, dataset_path=dataset_path, use_gpu=False, generate_annotate=False)
	# report.generate()
	report.generate_graphs_loss_history()
