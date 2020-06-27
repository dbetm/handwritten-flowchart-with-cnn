# -*- coding: utf-8 -*-

"""
Shape model module contains a class to manage the model for shapes and connectors
recognition, allows train an architecture that use CNNs and measure the same.
"""

import os
import json

from train import Trainer
from report import Report

class ShapeModel(object):
	"""ShapeModel allows to start and manager the training and test process
	in a deep learning arquitecture for object detection, such is Faster R-CNN.
	The implementation is based in code from: https://github.com/kbardool/keras-frcnn
	Mirror link: https://github.com/dbetm/keras-frcnn
	"""

	def __init__(self, dataset_path, num_rois=32, weights_input_path="none"):

		super(ShapeModel, self).__init__()

		self.dataset_path = dataset_path
		self.num_rois = num_rois
		self.weights_input_path = weights_input_path

	def __generate_results_path(self, base):

		ans = base + "_results"
		folder = os.listdir(ans)
		num_results = len(folder)

		name = ans + "/" + str(num_results)
		while(os.path.isdir(name)):
			num_results += 1
			name = ans + "/" + str(num_results)

		return name

	def train(
			self,
			data_augmentation,
			num_epochs=5,
			epoch_length=32,
			learning_rate=1e-5,
			num_rois=32,
			use_gpu=False,
	):
		"""Fit deep learning model."""

		# Initialize paths when creating the results folder
		base_path = self.__generate_results_path("training")
		annotate_path = base_path + "/annotate.txt"
		weights_output_path = base_path + "/flowchart_3b_model.hdf5"
		config_output_filename = base_path + "/config.pickle"
		# Create folder training folder
		os.mkdir(base_path)
		# Instance Trainer
		trainer = Trainer(base_path, use_gpu)
		# Recover data from dataset
		trainer.recover_data(
			self.dataset_path,
			annotate_path,
			generate_annotate=True
		)
		# Configure trainer
		trainer.configure(
			data_augmentation,
			self.num_rois,
			weights_output_path,
			self.weights_input_path,
			num_epochs=num_epochs,
			epoch_length=epoch_length,
			learning_rate=learning_rate,
		)
		trainer.save_config(config_output_filename)
		trainer.train()

	def generate_classification_report(
		self,
		results_path,
		generate_annotate=False,
		use_gpu=False
		):
		"""Generate classification report with model pre-trained.
			- Calculate mAP (mean Average Precision)
			- Generate classification report.
			- Generate confusion matrix.
		"""

		report = Report(
			results_path=results_path,
			dataset_path=self.dataset_path,
			generate_annotate=generate_annotate,
			use_gpu=use_gpu
		)
		report.generate()


def get_options():
	"""Util function to load options for training process."""
	try:
		with open('args.json', 'r') as f:
			options_dict = json.load(f)
	except Exception as e:
		print("Options file (JSON) don't found!")
		exit()

	return options_dict


if __name__ == '__main__':
	"""Please, if you want to generate the report for a specified model,
	uncomment the last block of code and comment first lines.
	"""
	options_dict = get_options()
	print(options_dict)

	# Set default values
	if(options_dict['rois'] == None):
		options_dict['rois'] = 32
	if(options_dict['input_weight_path'] == None):
		options_dict['input_weight_path'] = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
	if(options_dict['epochs'] == None):
		options_dict['epochs'] = 5
	if(options_dict['learning_rate'] == None):
		options_dict['learning_rate'] = 1e-5

	shape_model = ShapeModel(
		dataset_path=options_dict['dataset'],
		num_rois=options_dict['rois'],
		weights_input_path=options_dict['input_weight_path']
		# weights_input_path="training_results/1/flowchart_3b_model.hdf5"
		# weights_input_path="vgg16_weights_tf_dim_ordering_tf_kernels.h5"
	)
	# testing train
	shape_model.train(
	    data_augmentation=True,
	    num_epochs=options_dict['epochs'],
		learning_rate=options_dict['learning_rate'],
		use_gpu=options_dict['gpu']
	)

	# -!-!- Temp: Unit testing for training without integration -!-!-

	# shape_model = ShapeModel(
	# 	dataset_path="/home/david/Escritorio/flowchart_3b_v3.1",
	# 	num_rois=32,
	# 	# weights_input_path="training_results/1/flowchart_3b_model.hdf5"
	# 	weights_input_path="vgg16_weights_tf_dim_ordering_tf_kernels.h5"
	# )
	#testing train
	# shape_model.train(
	#     data_augmentation=True,
	#     num_epochs=1,
	# 	learning_rate=1e-5,
	# 	use_gpu=False
	# )

	# shape_model.generate_classification_report(
	# 	results_path = "training_results/5/",
	# 	generate_annotate=False
	# )
