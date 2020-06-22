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
