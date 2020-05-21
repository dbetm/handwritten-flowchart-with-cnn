# -*- coding: utf-8 -*-

import copy
import cv2
import numpy as np
import imutils
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

		if augment and self.config.data_augmentation and np.random.randint(0, 2) == 0:
			rows, cols = img.shape[:2]
			option = np.random.randint(0, 3)
			if(option == 0):
				delta = np.random.randint(1, 70)
				contrast = np.random.randint(-delta, delta)
				brightness = np.random.randint(-delta, delta)
				img = np.int16(img)
				img = img * (contrast/127+1) - contrast + brightness
				img = np.clip(img, 0, 255)
				img = np.uint8(img)
			elif(option == 1):
				correction = 0.5
				# invGamma = 1.0 / correction
				img = img / 255.0
				img = cv2.pow(img, correction)
				img = np.uint8(img * 255)
			else:
				angle = 1 if(np.random.randint(0,2) == 1) else -1
				img = imutils.rotate_bound(img, angle)

		img_data_aug['width'] = img.shape[1]
		img_data_aug['height'] = img.shape[0]

		return img_data_aug, img

	def __verify_data(self, img_data):
		"""Verify full data in img_data"""

		assert 'filepath' in img_data
		assert 'bboxes' in img_data
		assert 'width' in img_data
		assert 'height' in img_data
