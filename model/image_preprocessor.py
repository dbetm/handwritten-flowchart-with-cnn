# -*- coding: utf-8 -*-

import os
import copy

import cv2
import numpy as np


class ImagePreprocessor(object):
    """Manipulate digital images to improve the handwritten flowcharts
    photos.
    """

    def __init__(self, image):
        super(ImagePreprocessor, self).__init__()
        self.image = image

    def to_gray_scales(self):
        """Return the image in gray scales."""

        res = copy.deepcopy(self.image)
        # Converting color image to grayscale image
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        return res

    def adaptative_thresholding_otsu(self):
        """Return thresholdized image, convert image to grayscale, then
        apply a Gaussian filter, finally apply Otsu's thresholding.
        """

        img = self.to_gray_scales()
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5,5), 0)
        retV, res = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        return res

    # Best visual results
    def apply_unsharp_masking(self):
        """
        """
        img = self.to_gray_scales()
        #img = copy.deepcopy(self.image)
        # Gaussian filtering
        blur = cv2.GaussianBlur(img, (5,5), 0)
        alpha = 1.5
        beta = 1.0 - alpha
        res = cv2.addWeighted(img, alpha, blur, beta, 0.0)

        return res

    def adaptative_thresholding(self):
        """Return thresholdized image, convert image to grayscale, then
        apply adaptive thresh mean, C = 8.
        """

        img = self.to_gray_scales()
        res = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 8
        )
        return res

    def adaptative_thresholding_gaussian(self):
        """Return thresholdized image, convert image to grayscale, then
        apply adaptive thresh Gaussian, C = 7.
        """

        img = self.to_gray_scales()
        res = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 7
        )
        return res

    def augment_brightness(self, delta):

        img = self.to_gray_scales()
        contrast = 0
        brightness = 10 if abs(delta) >= 255 else delta
        img = np.int16(img)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        return img

path = "/home/david/Escritorio/set10/"
res_folder = "testing_img_prep"
files = os.listdir(path)

for file in files:
    print(file)
    ip = ImagePreprocessor(cv2.imread(path + "/" + file))
    #cv2.imwrite(res_folder + "/" + file, ip.adaptative_thresholding_otsu())
    #cv2.imwrite(res_folder + "/" + file, ip.adaptative_thresholding_gaussian())
    #cv2.imwrite(res_folder + "/" + file, ip.augment_brightness(20))
    cv2.imwrite(res_folder + "/" + file, ip.apply_unsharp_masking())

