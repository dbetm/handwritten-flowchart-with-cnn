# -*- coding: utf-8 -*-

from parser import Parser
import random
import cv2

from image_tools import ImageTools


class Checker(object):
	"""Check all bounding boxes each image, and draw the them."""

	def __init__(self, dataset_path, annotate_path):
		super(Checker, self).__init__()
		self.dataset_path = dataset_path
		self.annotate_path = annotate_path
		self.all_imgs = []
		self.__load_data()


	def __load_data(self):
		parser = Parser(
			dataset_path=dataset_path,
			annotate_path=self.annotate_path,
		)
		# Recover image paths
		self.all_imgs, _, _ = parser.get_data(generate_annotate=False)
		# random.shuffle(self.all_imgs)

	def show_samples(self, num_imgs):
		"""Drawing and show some sample images."""
		train_images = [s for s in self.all_imgs if s['imageset'] == 'trainval']
		num_imgs = num_imgs if num_imgs < len(train_images) else len(train_images)
		random.shuffle(train_images)
		some_imgs = train_images[:num_imgs]
		print("."*45)

		for img in some_imgs:
			path = img['filepath']
			image = cv2.imread(path)
			width = img['width']
			height = img['height']
			# print(width)
			# print(height)
			bboxes = img['bboxes']

			for bbox in bboxes:
				_class = bbox['class']
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				# rectangle shape
				cv2.rectangle(image, (x1, y1), (x2, y2), (170, 30, 5), 3)
				# text
				(retval, baseLine) = cv2.getTextSize(
					_class,
					cv2.FONT_HERSHEY_SIMPLEX,
					1,
					1
				)
				textOrg = (x1, y1)
				# Rectangle for text
				cv2.rectangle(
					image,
					(textOrg[0] - 5, textOrg[1] + baseLine - 5),
					(textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
					(22, 166, 184),
					2
				)
				# Fill text rectangle
				cv2.rectangle(
					image,
					(textOrg[0] - 5, textOrg[1] + baseLine - 5),
					(textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
					(255, 255, 255),
					-1
				)
				# Put class text
				cv2.putText(
					image,
					_class,
					textOrg,
					cv2.FONT_HERSHEY_DUPLEX,
					1,
					(0, 0, 0),
					1
				)

			(width, height) = ImageTools.get_new_img_size(width, height)
			image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
			cv2.imshow('sample', image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()


if __name__ == '__main__':
	dataset_path = "/home/david/Escritorio/flowchart_3b_v3"
	annotate_path = "/home/david/Escritorio/handwritten-flowchart-with-cnn/model/frcnn/utilities/annotate.txt"
	checker = Checker(
		dataset_path=dataset_path,
		annotate_path=annotate_path
	)

	checker.show_samples(80)
