# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET


class Parser(object):
	"""Recovery the data from annotations file or generate annotations file
	from dataset.
	"""

	def __init__(self, dataset_path, annotate_path="annotate.txt"):
		super(Parser, self).__init__()

		self.dataset_path = dataset_path
		self.annotate_path = annotate_path
		# Dictionaries for save the data
		self.all_imgs = {}
		self.classes_count = {}
		self.class_mapping = {}
		# other attribs
		self.found_bg = False

	def get_data(self, generate_annotate=False):
		"""Parsing annotations files."""

		if(generate_annotate):
			self.__generate_annotate()
		# Open annotations file
		with open(self.annotate_path, 'r') as f:
			print('Parsing annotation file...')
			for line in f:
				self.__read_line(line)
			all_data = []
			for key in self.all_imgs:
				all_data.append(self.all_imgs[key])

			# make sure the bg class is last in the list
			if self.found_bg:
				length = len(self.class_mapping)
				if self.class_mapping['bg'] != length - 1:
					key_to_switch = [key for key in self.class_mapping.keys() if self.class_mapping[key] == length-1][0]
					val_to_switch = self.class_mapping['bg']
					self.class_mapping['bg'] = length - 1
					self.class_mapping[key_to_switch] = val_to_switch
			f.close()
			return all_data, self.classes_count, self.class_mapping

	def __read_line(self, line):
		"""Read the next line in annotations file."""

		line_split = line.strip().split(',')
		(filename, x1, y1, x2, y2, class_name, test) = line_split
		test = (test == "1")

		if class_name not in self.classes_count:
			self.classes_count[class_name] = 1
		else:
			self.classes_count[class_name] += 1

		if class_name not in self.class_mapping:
			if class_name == 'bg' and self.found_bg == False:
				print('Found class name with special name bg. Will be treated as a background region.')
				self.found_bg = True
			self.class_mapping[class_name] = len(self.class_mapping)

		if filename not in self.all_imgs:
			self.all_imgs[filename] = {}
			rows = None
			cols = None
			try:
				img = cv2.imread(filename)
				(rows,cols) = img.shape[:2]
			except Exception as e:
				print(e)
				print("It is suggested to generate the annotation file again.")
				exit()

			self.all_imgs[filename]['filepath'] = filename
			self.all_imgs[filename]['width'] = cols
			self.all_imgs[filename]['height'] = rows
			self.all_imgs[filename]['bboxes'] = []
			if(test == True):
				self.all_imgs[filename]['imageset'] = 'test'
			else:
				self.all_imgs[filename]['imageset'] = 'trainval'

		new_image_data = {
			'class': class_name,
			'x1': int(x1),
			'x2': int(x2),
			'y1': int(y1),
			'y2': int(y2)
		}
		self.all_imgs[filename]['bboxes'].append(new_image_data)

	def __generate_annotate(self):
		"""Generate annotation file from dataset."""

		if not(os.path.isdir(self.dataset_path)):
			print("Error: Not valid dataset path!")
			exit()
		print("Generating annots file...")

		train_path = self.dataset_path + "/train"
		test_path = self.dataset_path + "/validation"

		data = self.__collect_data(train_path, test=False, data=[])
		data = self.__collect_data(test_path, test=True, data=data)

		data = pd.DataFrame(data)
		data.to_csv(self.annotate_path, header=None, index=None, sep=',')

	def __collect_data(self, path, test=False, data=[]):
		"""Get data from validation or training folder of dataset (xml files)."""

		# Check if it will get from validation or train
		if(test):
			type = 1
			base_path = self.dataset_path + "/validation/"
		else:
			type = 0
			base_path = self.dataset_path + "/train/"

		path_dataset = os.listdir(path)
		# Get for each class folder
		for class_i in path_dataset:
			items_path = path + "/" + class_i + "/annots"
			base = base_path + class_i + "/images"
			items = os.listdir(items_path)
			items.sort(key=Parser.get_int)
			for item in items:
				row = []
				parsedXML = ET.parse(items_path + "/" + item)
				image_path = base + "/" + (item.split(".")[0]) + ".jpg"
				for node in parsedXML.getroot().iter('object'):
					xmin = int(node.find('bndbox/xmin').text)
					ymin = int(node.find('bndbox/ymin').text)
					xmax = int(node.find('bndbox/xmax').text)
					ymax = int(node.find('bndbox/ymax').text)
					_class = node.find('name').text
					row = [image_path, xmin, ymin, xmax, ymax, _class, type]
					data.append(row)

		return data

	@staticmethod
	def get_int(name):
		"""Convert a numeric filename in integer."""

		num, extension = name.split('.')

		return int(num)

if __name__ == '__main__':
	simple_parser = Parser(
		dataset_path="/home/david/datasets/flowchart-3b(splitter)",
		annotate_path="annotate.txt"
	)
	all_data, classes_count, class_mapping = simple_parser.get_data(
		generate_annotate=False
	)
	print(all_data)
	print("-"*35)
	print(classes_count)
	print("-"*35)
	print(class_mapping)
