#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import logging

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils

from frcnn.data_generator import Metrics, Utils
from frcnn.losses import LossesCalculator
from frcnn.roi_helpers import ROIHelpers
from frcnn.cnn import CNN
from frcnn.utilities.config import Config
from frcnn.utilities.parser import Parser

class Trainer(object):
	"""Setup training and run for some epochs."""

	def __init__(self):
		super(Trainer, self).__init__()
		self.config = Config()
		self.__setup()
		self.parser = None
		self.all_data = []
		self.classes_count = []
		self.class_mapping = []
		self.num_images = 0
		self.num_anchors = 0
		self.input_shape_image = None
		# Datasets for training, split 80% training and 20% for validation
		self.train_images = None
		self.val_images = None
		# Convolutional Neural Network
		self.cnn = None
		# Data generators
		self.data_gen_train = None
		self.data_gen_val = None
		# Input Tensor Regions of Interest
		self.roi_input = Input(shape=(None, 4))
		# Models for Faster R-CNN
		self.model_rpn = None
		self.model_classifier = None
		self.model_all = None
		# Training process
		self.iter_num = 0
		self.losses = None
		self.rpn_accuracy_rpn_monitor = None
		self.rpn_accuracy_for_epoch = None

	def __setup(self):
		sys.setrecursionlimit(40000)
		logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

	def configure(self,
		horizontal_flips,
		vertical_flips,
		output_weight_path,
		num_rois,
		input_weight_path,
		num_epochs=5,
		epoch_length=32
		):
		self.config.horizontal_flips = horizontal_flips
		self.config.vertical_flips = vertical_flips
		self.config.model_path = output_weight_path
		self.config.num_rois = num_rois
		self.config.base_net_weights = input_weight_path
		self.config.num_epochs = num_epochs
		self.config.epoch_length = epoch_length

		self.num_anchors = len(self.config.anchor_box_scales)
		self.num_anchors *= len(self.config.anchor_box_ratios)
		# Instance convolutional neural network
		self.cnn = CNN(
			self.num_anchors,
			(self.roi_input, self.config.num_rois),
			len(self.classes_count)
		)
		# Tensor for image in TensorFlow
		self.input_shape_image = (None, None, 3)

	def recover_data(self, input_path):
		"""Recover data from annotate file or create annotate file from dataset.
		"""
		self.parser = Parser(
			input_path,
			annotate_path="frcnn/utilities/annotate.txt"
		)
		ans = self.parser.get_data(generate_annotate=False)
		self.all_data, self.classes_count, self.class_mapping = ans
		# If bg was not added, it will be added to the data image dictionaries.
		if 'bg' not in self.classes_count:
			self.classes_count['bg'] = 0
			self.class_mapping['bg'] = len(self.class_mapping)
		# Mapping persistence in config object
		self.config.class_mapping = self.class_mapping

		self.show_info_data()

	def show_info_data(self):
		"""Show data that it will use for training."""
		print('Training images per class:')
		pprint.pprint(self.classes_count)
		print('Num classes (including bg) = {}'.format(len(self.classes_count)))

	def save_config(self, config_output_filename):
		self.config.config_output_filename = config_output_filename
		with open(config_output_filename, 'wb') as config_f:
			pickle.dump(self.config, config_f)
			message = 'Config has been written to {}, and can be loaded when testing to ensure correct results'
			print(message.format(config_output_filename))

	def train(self, learning_rate=1e-5):
		self.__pre_train()
		self.__build_frcnn(learning_rate)

		# Iterative process
		iter_num = 0
		best_loss = np.Inf

		# Invert key-value in classes dictionary
		class_mapping_inv = {v: k for k, v in self.class_mapping.items()}

		# Start iterative process
		print("Starting training :)")

		for epoch_num in range(self.config.num_epochs):
			start_time = time.time()
			progress_bar = generic_utils.Progbar(self.config.epoch_length)
			print('Epoch {}/{}'.format(epoch_num + 1, self.config.num_epochs))
			while True:
				try:
					# If an epoch is completed + allowed verbose, then
					# Print the average number of overlapping bboxes.
					len_rpn_acc_rpn_moni = len(self.rpn_accuracy_rpn_monitor)
					cond1 = len_rpn_acc_rpn_moni == self.config.epoch_length
					if cond1 and self.config.verbose:
						self.__print_average_bbxes()

					X, Y, img_data = next(self.data_gen_train)
					# calc loss for RPN
					loss_rpn = self.model_rpn.train_on_batch(X, Y)
					# pred with RPN
					pred_rpn = self.model_rpn.predict_on_batch(X)
					# Convert RPN to ROI
					roi_helper = ROIHelpers(self.config, overlap_thresh=0.9, max_boxes=300)
					roi = roi_helper.convert_rpn_to_roi(
						pred_rpn[0],
						pred_rpn[1],
						use_regr=True
					)

					# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
					X2, Y1, Y2, ious = roi_helper.calc_iou(
						roi,
						self.all_data,
						self.class_mapping
					)

					if X2 is None:
						self.rpn_accuracy_rpn_monitor.append(0)
						self.rpn_accuracy_for_epoch.append(0)
						continue

					# Get negatives samples and positive samples (IoU > thresh)
					neg_samples = np.where(Y1[0, :, -1] == 1)
					pos_samples = np.where(Y1[0, :, -1] == 0)
					neg_sample, pos_samples = self.__validate_samples(
						neg_samples,
						pos_samples
					)

					self.rpn_accuracy_rpn_monitor.append(len(pos_samples))
					self.rpn_accuracy_for_epoch.append((len(pos_samples)))

					# Select samples from positives and negatives samples
					sel_samples = self.__select_samples(neg_samples, pos_samples)

					# Update losses, for class detector and RPN
					self.__update_losses(sel_samples, iter_num, loss_rpn)
					# Update progress bar in an epoch
					progbar.update(
						iter_num+1,
						[
							('rpn_cls', self.losses[iter_num, 0]),
							('rpn_regr', self.losses[iter_num, 1]),
							('detec_cls', self.losses[iter_num, 2]),
							('detec_regr', self.losses[iter_num, 3])
						]
				    )

					iter_num += 1

					logging.debug("All is well")
					# If the epoch actual is completed
					if iter_num == self.config.epoch_length:
						best_loss = self.__update_losses_in_epoch(best_loss, start_time)
						iter_num = 0
						break

				except Exception as e:
					print('Exception: {}'.format(e))
					#continue
					break

		print('Training complete, exiting :p.')

	def __pre_train(self):
		# Randomize data
		random.shuffle(self.all_data)
		# Set for training process
		self.num_images = len(self.all_data)
		self.train_images = [s for s in self.all_data if s['imageset'] == 'trainval']
		self.val_images = [s for s in self.all_data if s['imageset'] == 'test']
		print('Num train samples {}'.format(len(self.train_images)))
		print('Num val samples {}'.format(len(self.val_images)))
		# Create data generators
		self.data_gen_train = Utils.get_anchor_gt(
			self.train_images,
			self.classes_count,
			self.config,
			CNN.get_img_output_length,
			mode='train'
		)
		self.data_gen_val = Utils.get_anchor_gt(
			self.val_images,
			self.classes_count,
			self.config,
			CNN.get_img_output_length,
			mode='val'
		)

		self.losses = np.zeros((self.config.epoch_length, 5))
		self.rpn_accuracy_rpn_monitor = []
		self.rpn_accuracy_for_epoch = []

	def __build_frcnn(self, learning_rate):
		img_input = Input(shape=self.input_shape_image)
		# Define the base network (VGG16)
		shared_layers = self.cnn.build_nn_base(img_input)
		# Define the RPN, built on the base layers
		rpn = self.cnn.create_rpn(shared_layers)

		classifier = self.cnn.build_classifier(
			shared_layers,
			num_classes=len(self.classes_count)
		)
		# Build models for Faster R-CNN
		self.model_rpn = Model(img_input, rpn[:2])
		self.model_classifier = Model([img_input, self.roi_input], classifier)

		# This is a model that holds both the RPN and the classifier...
		# used to load/save weights for the models.
		self.model_all = Model([img_input, self.roi_input], rpn[:2] + classifier)

		self.__load_weights()
		self.__compile_models(learning_rate)

	def __compile_models(self, learning_rate):
		# Create optimizers and compile models
		num_classes = len(self.classes_count)
		losses = LossesCalculator(num_classes, self.num_anchors)

		optimizer = Adam(lr=learning_rate)
		optimizer_classifier = Adam(lr=learning_rate)
		logging.debug("Compile model_rpn") # DEBUG
		self.model_rpn.compile(
			optimizer=optimizer,
			loss=[
				LossesCalculator.rpn_loss_cls(),
				LossesCalculator.rpn_loss_regr()
			],
		)
		logging.debug("Compile model_classifier") # DEBUG
		self.model_classifier.compile(
			optimizer=optimizer_classifier,
			loss=[
				LossesCalculator.class_loss_cls,
				LossesCalculator.class_loss_regr()
			],
			metrics={'dense_class_{}'.format(num_classes): 'accuracy'},
		)
		logging.debug("Compile model_all") # DEBUG
		self.model_all.compile(
			optimizer='sgd',
			loss='mae' # Mean Absolute Error
		)

	def __load_weights(self):
		try:
			print('Loading weights from {}'.format(self.config.base_net_weights))
			self.model_rpn.load_weights(self.config.base_net_weights, by_name=True)
			self.model_classifier.load_weights(
				self.config.base_net_weights,
				by_name=True
			)
		except Exception as e:
			print('Exception: {}'.format(e))
			print("Couldn't  load pretrained model weights.")
			print("Weights can be found in the keras application folder \
				https://github.com/fchollet/keras/tree/master/keras/applications")

	def __print_average_bbxes():
		"""Show the average number of overlapping bboxes."""
		sum = sum(self.rpn_accuracy_rpn_monitor)
		mean_overlapping_bboxes = float(sum) / len(self.rpn_accuracy_rpn_monitor)
		self.rpn_accuracy_rpn_monitor = []
		message = "Average number of overlapping bounding boxes from RPN = {} for {} previous iteration(s)."
		print(message.format(mean_overlapping_bboxes, self.config.epoch_length))
		if mean_overlapping_bboxes == 0:
			message = "RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training."
			print(message)

	def __validate_samples(self, neg_samples, pos_samples):
		if len(neg_samples) > 0:
			# Just choose the first one
			neg_samples = neg_samples[0]
		else:
			# Leave the negative samples list empty
			neg_samples = []
		if len(pos_samples) > 0:
			pos_samples = pos_samples[0]
		else:
			pos_samples = []

		return neg_samples, pos_samples

	def __select_samples(self, neg_samples, pos_samples):
		if self.config.num_rois > 1:
			if len(pos_samples) < self.config.num_rois // 2:
				selected_pos_samples = pos_samples.tolist()
			else:
				selected_pos_samples = np.random.choice(
					pos_samples,
					self.config.num_rois // 2,
					replace=False
				).tolist()
			try:
				selected_neg_samples = np.random.choice(
					neg_samples,
					self.config.num_rois - len(selected_pos_samples),
					replace=False
				).tolist()
			except:
				"""The replace parameter determines whether or not the selection
				is made with replacement (default this parameter takes
				the value False).
				"""
				selected_neg_samples = np.random.choice(
					neg_samples,
					self.config.num_rois - len(selected_pos_samples),
					replace=True
				).tolist()

			sel_samples = selected_pos_samples + selected_neg_samples
		else:
			# in the extreme case where num_rois = 1
			# we pick a random pos or neg sample
			selected_pos_samples = pos_samples.tolist()
			selected_neg_samples = neg_samples.tolist()
			if np.random.randint(0, 2):
				sel_samples = random.choice(neg_samples)
			else:
				sel_samples = random.choice(pos_samples)

		return sel_samples

	def __update_losses(self, sel_samples, iter_num, loss_rpn):
		loss_class = self.model_classifier.train_on_batch(
			[X, X2[:, sel_samples, :]],
			[Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
		)

		self.losses[iter_num, 0] = loss_rpn[1]
		self.losses[iter_num, 1] = loss_rpn[2]

		self.losses[iter_num, 2] = loss_class[1]
		self.losses[iter_num, 3] = loss_class[2]
		self.losses[iter_num, 4] = loss_class[3]

	def __update_losses_in_epoch(self, best_loss, start_time):
		loss_rpn_cls = np.mean(self.losses[:, 0])
		loss_rpn_regr = np.mean(self.losses[:, 1])
		loss_class_cls = np.mean(self.losses[:, 2])
		loss_class_regr = np.mean(self.losses[:, 3])
		class_acc = np.mean(self.losses[:, 4])

		sum = sum(self.rpn_accuracy_for_epoch)
		mean_overlapping_bboxes = float(sum) / len(self.rpn_accuracy_for_epoch)
		self.rpn_accuracy_for_epoch = []

		if self.config.verbose:
			message = 'Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'
			print(message.format(mean_overlapping_bboxes))
			message = 'Classifier accuracy for bounding boxes from RPN: {}'
			print(message.format(class_acc))
			print('Loss RPN classifier: {}'.format(loss_rpn_cls))
			print('Loss RPN regression: {}'.format(loss_rpn_regr))
			print('Loss Detector classifier: {}'.format(loss_class_cls))
			print('Loss Detector regression: {}'.format(loss_class_regr))
			print('Elapsed time: {}'.format(time.time() - start_time))

		curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

		if curr_loss < best_loss:
			if self.config.verbose:
				message = 'Total loss decreased from {} to {}, saving weights'
				print(message.format(best_loss, curr_loss))
			best_loss = curr_loss
			self.model_all.save_weights(self.config.model_path)

		return best_loss

if __name__ == '__main__':
	trainer = Trainer()
	input_path_weights = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
	trainer.recover_data("/home/david/datasets/flowchart-3b(splitter)")
	trainer.configure(False, False, "model_frcnn_v0.hdf5", 32, input_path_weights)
	trainer.save_config("config.pickle")
	trainer.train(learning_rate=1e-5)
