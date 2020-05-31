import cv2
import numpy as np
import copy
import imutils

class DataAugmentation(object):
	"""docstring for DataAugmentation."""

	def __init__(self, image_path):
		super(DataAugmentation, self).__init__()
		self.image = cv2.imread(image_path)

	def change_brightness_and_contrast(self):
		new_img = copy.deepcopy(self.image)

		delta = np.random.randint(1, 70)
		contrast = np.random.randint(-delta, delta)
		brightness = np.random.randint(-delta, delta)
		new_img = np.int16(new_img)
		new_img = new_img * (contrast/127+1) - contrast + brightness
		new_img = np.clip(new_img, 0, 255)
		new_img = np.uint8(new_img)

		return new_img

	def apply_gamma_correction(self):
		new_img = copy.deepcopy(self.image)

		correction = 0.5
		# invGamma = 1.0 / correction
		new_img = new_img / 255.0
		new_img = cv2.pow(new_img, correction)
		return np.uint8(new_img * 255)

	def rotate_bit(self):
		new_img = copy.deepcopy(self.image)
		angle = 1 if(np.random.randint(0,2) == 1) else -1
		new_img = imutils.rotate_bound(new_img, angle)

		return new_img

	@staticmethod
	def rotate_a_bit(img):
		angle = 4 if(np.random.randint(0,2) == 1) else -4
		img = imutils.rotate_bound(img, angle)


	def tilt(self):
		src = copy.deepcopy(self.image)

		# srcTri = np.array(
		#     [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]]
		# ).astype(np.float32)
		src_points = np.array([
			[0,0],
			[src.shape[1]-1, 0],
			[src.shape[1]-1, src.shape[0]-1]
		]).astype(np.float32)

		# dstTri = np.array(
		#     [[0, src.shape[1]*0.2], [src.shape[1]*0.55, src.shape[0]*0.25],
		#     [src.shape[1]*0.15, src.shape[0]*0.5]
		# ).astype(np.float32)
		dest_src_points = np.array([
			[src.shape[1]*0.04,0],
			[src.shape[1]-1, 0],
			[src.shape[1]*0.96, src.shape[0]-1]
		]).astype(np.float32)
		warp_mat = cv2.getAffineTransform(src_points, dest_src_points)
		warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

		return warp_dst


	def to_gray_scales(self):
		"""Return the image in gray scales."""

		res = copy.deepcopy(self.image)
		# Converting color image to grayscale image
		res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		return res

	def skeletonization(self):
		# No
		img = self.to_gray_scales()
		skeleton = imutils.skeletonize(img, size=(5, 5))

		return skeleton

	def add_gaussian_noise(self):
		"""Poisson"""
		img = copy.deepcopy(self.image)
		# row,col,ch = img.shape
		# gauss = np.random.randn(row,col,ch)
		# gauss = gauss.reshape(row,col,ch)
		# noisy = img + img * gauss
		# return noisy
		vals = len(np.unique(img))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(img * vals) / float(vals)
		return noisy

image_path = "/home/david/Escritorio/set1/1.jpg"
daug = DataAugmentation(image_path)

# for i in range(5):
#     name = str(i) + "sample.jpg"
#     #cv2.imwrite(name, daug.change_brightness_and_contrast())
#     #cv2.imwrite(name, daug.apply_gamma_correction())
#     cv2.imwrite(name, daug.rotate_bit())
#image = cv2.imread(image_path)
#DataAugmentation.rotate_a_bit(image)
cv2.imwrite("sampling.jpg", daug.tilt())
