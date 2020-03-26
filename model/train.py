# -*- coding: utf-8 -*-

from __future__ import division
import random
import pprint
import sys
import time
import pickle
import logging
import traceback
from optparse import OptionParser

import numpy as np
import tensorflow as tf
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
from frcnn.utilities.history import History


class Trainer(object):
	"""Setup training and run for some epochs."""

	def __init__(self, results_path, use_gpu=False):
		super(Trainer, self).__init__()

		self.config = Config()
		self.config.use_gpu = use_gpu
		self.parser = None
		self.all_data = []
		self.classes_count = []
		self.class_mapping = []
		self.num_images = 0
		self.num_anchors = 0
		self.input_shape_image = None
		self.results_path = results_path
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
		self.history = History(results_path)
		# System and session setup
		self.__setup()

	def __setup(self):
		"""System and session, setup."""

		sys.setrecursionlimit(40000)
		logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
		if(self.config.use_gpu):
			config_gpu = tf.compat.v1.ConfigProto()
			# dynamically grow the memory used on the GPU
			config_gpu.gpu_options.allow_growth = True
			# to log device placement (on which device the operation ran)
			config_gpu.log_device_placement = True
			sess = tf.compat.v1.Session(config=config_gpu)

	def configure(
			self,
			horizontal_flips,
			vertical_flips,
			num_rois,
			weights_output_path,
			weights_input_path,
			num_epochs=5,
			epoch_length=32,
			learning_rate=1e-5):
		"""Set hyperparameters before the training process."""

		# Config file
		self.config.horizontal_flips = horizontal_flips
		self.config.vertical_flips = vertical_flips
		self.config.num_rois = num_rois
		self.config.weights_output_path = weights_output_path
		self.config.weights_input_path = weights_input_path
		self.config.num_epochs = num_epochs
		self.config.epoch_length = epoch_length
		self.config.learning_rate = learning_rate
		# Trainer
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

	def recover_data(
			self,
			dataset_path,
			annotate_path="frcnn/utilities/annotate.txt",
			generate_annotate=False):
		"""Recover data from annotate file or create annotate file from dataset.
		"""
		# Instance parser, recover data from annotate file or dataset
		self.parser = Parser(
			dataset_path=dataset_path,
			annotate_path=annotate_path
		)
		# Get data dictionaries
		ans = self.parser.get_data(generate_annotate=generate_annotate)
		self.all_data, self.classes_count, self.class_mapping = ans
		# If bg was not added, it will be added to the image data dictionaries.
		if 'bg' not in self.classes_count:
			self.classes_count['bg'] = 0
			self.class_mapping['bg'] = len(self.class_mapping)
		# Mapping persistence in config object
		self.config.class_mapping = self.class_mapping
		# Show resume from loaded data
		self.show_info_data()

	def show_info_data(self):
		"""Show data that it will use for training."""

		print('Training images per class:')
		pprint.pprint(self.classes_count)
		print('Num classes (including bg) = {}'.format(len(self.classes_count)))
		# Persistence the data
		self.history.save_classes_info(self.classes_count)

	def save_config(self, config_output_filename):
		"""Do persistence the config data for training process."""

		self.config.config_output_filename = config_output_filename
		with open(config_output_filename, 'wb') as config_f:
			pickle.dump(self.config, config_f)
			message = 'Config has been written to {}, and can be '
			message += 'loaded when testing to ensure correct results'
			print(message.format(config_output_filename))

	def train(self):
		"""Train the Faster R-CNN."""

		self.__prepare_train()
		self.__build_frcnn()

		# Iterative process
		iter_num = 0
		best_loss = np.Inf

		# Start iterative process
		print("The training has begun :)")
		for epoch_num in range(self.config.num_epochs):
			start_time = time.time() # init time for current epoch
			# Instance progress bar for display progress in current epoch
			progress_bar = generic_utils.Progbar(self.config.epoch_length)
			print('Epoch {}/{}'.format(epoch_num + 1, self.config.num_epochs))

			while True:
				try:
					# If an epoch is completed + allowed verbose, then:
					# print the average number of overlapping bboxes.
					len_rpn_acc_rpn_moni = len(self.rpn_accuracy_rpn_monitor)
					cond1 = (len_rpn_acc_rpn_moni == self.config.epoch_length)
					if cond1 and self.config.verbose:
						self.__print_average_bbxes()

					X, Y, img_data = next(self.data_gen_train)
					# calc loss for RPN
					loss_rpn = self.model_rpn.train_on_batch(X, Y)
					# pred with RPN
					pred_rpn = self.model_rpn.predict_on_batch(X)
					# Instance a ROI Helper
					roi_helper = ROIHelpers(
						self.config,
						overlap_thresh=0.9,
						max_boxes=300
					)
					# Convert RPN to ROI
					roi = roi_helper.convert_rpn_to_roi(
						pred_rpn[0],
						pred_rpn[1],
						use_regr=True
					)
					# Calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
					X2, Y1, Y2, ious = roi_helper.calc_iou(
						roi,
						img_data,
						self.class_mapping
					)

					if X2 is None:
						self.rpn_accuracy_rpn_monitor.append(0)
						self.rpn_accuracy_for_epoch.append(0)
						continue

					# Get negatives samples and positive samples (IoU > thresh)
					neg_samples = np.where(Y1[0, :, -1] == 1)
					pos_samples = np.where(Y1[0, :, -1] == 0)

					neg_samples, pos_samples = self.__validate_samples(
						neg_samples,
						pos_samples
					)

					self.rpn_accuracy_rpn_monitor.append(len(pos_samples))
					self.rpn_accuracy_for_epoch.append((len(pos_samples)))

					# Select samples from positives and negatives samples
					sel_samples = self.__select_samples(neg_samples, pos_samples)
					# Update losses, for class detector and RPN
					self.__update_losses(sel_samples, iter_num, loss_rpn, X, X2, Y1, Y2)
					# Update progress bar in the current epoch
					progress_bar.update(
						iter_num + 1,
						[
							('rpn_cls', self.losses[iter_num, 0]),
							('rpn_regr', self.losses[iter_num, 1]),
							('det_cls', self.losses[iter_num, 2]),
							('det_regr', self.losses[iter_num, 3]),
							('epoch', int(epoch_num + 1))
						]
				    )

					iter_num += 1

					# If the current epoch is completed
					if iter_num == self.config.epoch_length:
						best_loss = self.__update_losses_in_epoch(
							epoch_num,
							best_loss,
							start_time
						)
						iter_num = 0
						break

				except Exception as e:
					#traceback.print_exc()
					print('Exception: {}'.format(e))
					continue

		print('Training complete!!!, exiting :p')

	def __prepare_train(self):
		"""Initialize data generators, shuffle the data and create other
		data structures.
		"""

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

	def __build_frcnn(self):
		"""Create the whole model of the Faster R-CNN."""

		img_input = Input(shape=self.input_shape_image)
		# Define the base network (VGG16)
		shared_layers = self.cnn.build_nn_base(img_input)
		# Define the RPN, built on the base layers.
		rpn = self.cnn.create_rpn(shared_layers)
		# Define classifier, it will assign the class of the detected objects.
		classifier = self.cnn.build_classifier(
			shared_layers,
			num_classes=len(self.classes_count)
		)
		# Build models for Faster R-CNN.
		self.model_rpn = Model(img_input, rpn[:2])
		self.model_classifier = Model([img_input, self.roi_input], classifier)
		# This is a model that holds both the RPN and the classifier...
		# Used to load/save weights for the models
		self.model_all = Model([img_input, self.roi_input], rpn[:2] + classifier)
		# Use to load/save weights for the models.
		self.__load_weights()
		# Save the models like a trainable object.
		self.__compile_models()

	def __compile_models(self):
		""" Create optimizers and compile models."""

		learning_rate = self.config.learning_rate

		num_classes = len(self.classes_count)
		losses = LossesCalculator(num_classes, self.num_anchors)

		optimizer = Adam(lr=learning_rate)
		optimizer_classifier = Adam(lr=learning_rate)

		self.model_rpn.compile(
			optimizer=optimizer,
			loss=[
				LossesCalculator.rpn_loss_cls(),
				LossesCalculator.rpn_loss_regr()
			]
		)

		self.model_classifier.compile(
			optimizer=optimizer_classifier,
			loss=[
				LossesCalculator.class_loss_cls,
				LossesCalculator.class_loss_regr()
			],
			metrics={'dense_class_{}'.format(num_classes): 'accuracy'},
		)

		self.model_all.compile(
			optimizer='sgd',
			loss='mae' # Mean Absolute Error
		)

		# test save summaries
		self.history.save_summary(self.model_rpn, "rpn")
		self.history.save_summary(self.model_classifier, "classifier")
		self.history.save_summary(self.model_all, "all")
		# test save plots
		self.history.save_model_image(self.model_rpn, "rpn")
		self.history.save_model_image(self.model_classifier, "classifier")
		self.history.save_model_image(self.model_all, "all")

	def __load_weights(self):
		"""Load weights from a pretrained model."""

		try:
			print('Loading weights from {}'.format(self.config.weights_input_path))
			self.model_rpn.load_weights(self.config.weights_input_path, by_name=True)
			self.model_classifier.load_weights(
				self.config.weights_input_path,
				by_name=True
			)
		except Exception as e:
			print('Exception: {}'.format(e))
			print("Couldn't  load pretrained model weights.")
			print("Weights can be found in the keras application folder \
				https://github.com/fchollet/keras/tree/master/keras/applications")

	def __print_average_bbxes(self):
		"""Show the average number of overlapping bboxes."""

		total = sum(self.rpn_accuracy_rpn_monitor)
		mean_overlapping_bboxes = float(total)
		mean_overlapping_bboxes /= len(self.rpn_accuracy_rpn_monitor)

		self.rpn_accuracy_rpn_monitor = []

		message = "Average number of overlapping bounding boxes from RPN = {}"
		message +=  " for {} previous iteration(s)."
		print(message.format(mean_overlapping_bboxes, self.config.epoch_length))

		if mean_overlapping_bboxes == 0:
			message = "RPN is not producing bounding boxes that overlap the "
			message += "ground truth boxes. Check RPN settings or keep training."
			print(message)

	def __validate_samples(self, neg_samples, pos_samples):
		"""Format positives and negatives samples."""

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

		return (neg_samples, pos_samples)

	def __select_samples(self, neg_samples, pos_samples):
		"""Select X positives samples and Y negatives samples for complete
		number RoIs.
		"""

		if self.config.num_rois > 1:
			if len(pos_samples) < self.config.num_rois // 2:
				selected_pos_samples = pos_samples.tolist()
			else:
				selected_pos_samples = np.random.choice(
					a=pos_samples,
					size=self.config.num_rois // 2,
					replace=False
				).tolist()
			try:
				selected_neg_samples = np.random.choice(
					a=neg_samples,
					size=self.config.num_rois - len(selected_pos_samples),
					replace=False
				).tolist()
			except:
				"""The replace parameter determines whether or not the selection
				is made with replacement (default this parameter takes
				the value False).
				"""
				selected_neg_samples = np.random.choice(
					a=neg_samples,
					size=self.config.num_rois - len(selected_pos_samples),
					replace=True
				).tolist()

			sel_samples = selected_pos_samples + selected_neg_samples
		else:
			"""In the extreme case where num_rois = 1, we pick a random pos
			or neg sample.
			"""
			selected_pos_samples = pos_samples.tolist()
			selected_neg_samples = neg_samples.tolist()
			if np.random.randint(0, 2):
				sel_samples = random.choice(neg_samples)
			else:
				sel_samples = random.choice(pos_samples)

		return sel_samples

	def __update_losses(self, sel_samples, iter_num, loss_rpn, X, X2, Y1, Y2):
		"""Update losses for RPN and classifier."""

		# Calculate weights according to classifier batch training.
		loss_class = self.model_classifier.train_on_batch(
			[X, X2[:, sel_samples, :]],
			[Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
		)

		self.losses[iter_num, 0] = loss_rpn[1]
		self.losses[iter_num, 1] = loss_rpn[2]
		self.losses[iter_num, 2] = loss_class[1]
		self.losses[iter_num, 3] = loss_class[2]
		self.losses[iter_num, 4] = loss_class[3]

	def __update_losses_in_epoch(self, epoch_num, best_loss, start_time):
		"""Update the final losses after the epochs ends."""

		# Average losses
		loss_rpn_cls = np.mean(self.losses[:, 0])
		loss_rpn_regr = np.mean(self.losses[:, 1])
		loss_class_cls = np.mean(self.losses[:, 2])
		loss_class_regr = np.mean(self.losses[:, 3])
		class_acc = np.mean(self.losses[:, 4])

		total = sum(self.rpn_accuracy_for_epoch)
		mean_overlapping_bboxes = float(total) / len(self.rpn_accuracy_for_epoch)
		total_time = time.time() - start_time
		self.rpn_accuracy_for_epoch = []
		# Print the resume of the epoch
		if self.config.verbose:
			message = 'Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'
			print(message.format(mean_overlapping_bboxes))
			message = 'Classifier accuracy for bounding boxes from RPN: {}'
			print(message.format(class_acc))
			print('Loss RPN classifier: {}'.format(loss_rpn_cls))
			print('Loss RPN regression: {}'.format(loss_rpn_regr))
			print('Loss detector classifier: {}'.format(loss_class_cls))
			print('Loss detector regression: {}'.format(loss_class_regr))
			print('Elapsed time: {}'.format(total_time))

		curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
		print('Best loss: {} vs current loss: {}'.format(best_loss, curr_loss))
		# Update the best loss if the current loss is better.
		if curr_loss < best_loss:
			message = 'Total loss decreased from {} to {}, saving weights'
			print(message.format(best_loss, curr_loss))
			best_loss = curr_loss
			# Save the best model
			self.history.save_best_model(
				self.model_all,
				self.config.weights_output_path
			)
		# Generate row for epoch info
		info = []
		# add data to info list
		info.append(epoch_num + 1)
		info.append(mean_overlapping_bboxes)
		info.append(class_acc)
		info.append(curr_loss)
		info.append(loss_rpn_cls)
		info.append(loss_rpn_regr)
		info.append(loss_class_cls)
		info.append(loss_class_regr)
		info.append(total_time)
		self.history.append_epoch_info(info)

		return best_loss

if __name__ == '__main__':
	results_path = "training_results/1"
	trainer = Trainer(results_path)
	weights_input_path = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
	path_dataset = "/home/david/Escritorio/flowchart-3b(splitter)"
	trainer.recover_data(
		path_dataset,
		generate_annotate=False,
		annotate_path=results_path + "/annotate.txt"
	)
	trainer.configure(
		horizontal_flips=False,
		vertical_flips=False,
		num_rois=32,
		weights_output_path=results_path + "/model_frcnn.hdf5",
		weights_input_path=weights_input_path,
		num_epochs=1
	)
	trainer.save_config(results_path + "/config.pickle")
	trainer.train()
