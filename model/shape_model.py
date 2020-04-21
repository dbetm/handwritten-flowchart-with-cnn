# -*- coding: utf-8 -*-

"""
Shape model module contains a class to manage the model for shapes and connectors
recognition, allows train an architecture that use CNNs and measure the same.
"""

__autor__ = "David"
__credits__ = ["David Betancourt Montellano", "Onder Francisco Campos García"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "davbetm@gmail.com"
__status__ = "Development"

import os

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

		return ans + "/" + str(num_results)

	def train(
			self,
			horizontal_flips,
			vertical_flips,
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
			horizontal_flips,
			vertical_flips,
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
		generate_annotate=False
		):
		"""Generate classification report with model pre-trained.
			- Calculate mAP (mean Average Precision)
			- Generate classification report.
			- Generate confusion matrix.
		"""

		report = Report(
			results_path=results_path,
			dataset_path=self.dataset_path,
			generate_annotate=generate_annotate
		)
		report.generate()


if __name__ == '__main__':
	shape_model = ShapeModel(
		dataset_path="/home/david/Escritorio/flowchart-3b(splitter)",
		num_rois=32,
		# weights_input_path="training_results/1/flowchart_3b_model.hdf5"
		# weights_input_path="vgg16_weights_tf_dim_ordering_tf_kernels.h5"
	)
	# testing train
	shape_model.train(
	    horizontal_flips=False,
	    vertical_flips=False,
	    num_epochs=5,
		learning_rate=0.1,
		use_gpu=True
	)

	# shape_model.generate_classification_report(
	# 	results_path = "training_results/x/",
	# 	generate_annotate=False
	# )
