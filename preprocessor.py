# -*- coding: utf-8 -*-

import os
import copy
import cv2
import numpy as np


class Preprocessor(object):
    """Class Preprocessor with utils for image preprocessing."""

    def __init__(self):
        super(Preprocessor, self).__init__()

    @staticmethod
    def to_gray_scale(image):
        """Return the image in gray scales."""

        # Converting color image to grayscale image
        res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return res

    @staticmethod
    def apply_unsharp_masking(image):
        """ Improve a bit the image for emphasize texture and details.
        """

        img = Preprocessor.to_gray_scale(image)

        # Gaussian filtering
        blur = cv2.GaussianBlur(img, (5,5), 0)
        alpha = 1.5
        beta = 1.0 - alpha
        res = cv2.addWeighted(img, alpha, blur, beta, 0.0)
        return res
    @staticmethod
    def resize_new_data(image,input_size):
        def get_max_min(image):
            h,w = image.shape
            argmin = float("inf")
            argmax = -float("inf")
            for i in range(h):
                for j in range(w):
                    if(image[i,j] == 0):
                        argmax = max(i,argmax)
                        argmin = min(i,argmin)
            return argmax,argmin
        def image_resize(image,height = None,inter = cv2.INTER_AREA):
            dim = None
            (h, w) = image.shape
            r = height / float(h)
            dim = (int(w * r),height)
            resized = cv2.resize(image,dim,interpolation = inter)
            return resized
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image,(3,3),0)
        ret,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        argmax,argmin = get_max_min(image)
        image = image[argmin:argmax]
        h,w = image.shape
        wt,ht = input_size
        image = illumination_compensation(image)
        image = remove_cursive_style(image)
        if argmax - argmin > input_size[1] // 2:
            image = image_resize(image,height = (input_size[1] // 2))
        h,w = image.shape
        target = np.ones((ht , wt), dtype=np.uint8)*255
        target[0:h,0:w] = image
        image = cv2.transpose(target)
        return image
