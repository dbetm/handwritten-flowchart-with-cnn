#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class ImageTools(object):
	"""Methods for get required format to the images."""

	def __init__(self):
		super(ImageTools, self).__init__()

	@staticmethod
	def get_new_img_size(width, height, img_min_side=600):
		"""Resize the image, based on a minimum side."""
		if width <= height:
			f = float(img_min_side) / width
			new_height = int(f * height)
			new_width = img_min_side
		else:
			f = float(img_min_side) / height
			new_width = int(f * width)
			new_height = img_min_side

		return new_width, new_height
