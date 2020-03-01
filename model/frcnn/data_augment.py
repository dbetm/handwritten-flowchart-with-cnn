# -*- coding: utf-8 -*-

import copy
import cv2
import numpy as np
import logging
import sys


class DataAugment(object):
	"""Generate variations in images-data"""

	def __init__(self, config):
		super(DataAugment, self).__init__()

		self.config = config
		logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

	def augment(self, img_data, augment=True):
		"""It does the variations defined in file config."""

		self.__verify_data(img_data)
		# Copy the data
		img_data_aug = copy.deepcopy(img_data)

		# Get the image from images folder
		img = cv2.imread(img_data_aug['filepath'])

		if augment:
			rows, cols = img.shape[:2]
			if self.config.use_horizontal_flips and np.random.randint(0, 2) == 0:
				img = cv2.flip(img, 1)
				for bbox in img_data_aug['bboxes']:
					x1 = bbox['x1']
					x2 = bbox['x2']
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
			if self.config.use_vertical_flips and np.random.randint(0, 2) == 0:
				img = cv2.flip(img, 0)
				for bbox in img_data_aug['bboxes']:
					y1 = bbox['y1']
					y2 = bbox['y2']
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2

		img_data_aug['width'] = img.shape[1]
		img_data_aug['height'] = img.shape[0]

		return img_data_aug, img

	def __verify_data(self, img_data):
		"""Verify full data in img_data"""

		assert 'filepath' in img_data
		assert 'bboxes' in img_data
		assert 'width' in img_data
		assert 'height' in img_data
