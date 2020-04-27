import cv2
import os
import numpy as np
from math import floor

class Preproc(object):
    def resize(self,image):
        h,w = image.shape
        ymin = float('inf')
        ymax = float('-inf')
        for x in range(w):
            for y in range(h):
                if(image[y,x] == 255):
                    ymin = min(y,ymin)
                    ymax = max(y,ymax)
        image = image[ymin:ymax,0:w]
        h,w = image.shape
        bg = None
        if(h > w):
            bg = np.zeros((h,h))
            for y in range(h):
                for x in range(w):
                    dif = floor((h - w)/2)
                    bg[y,x + dif] = image[y,x]
        elif(w > h):
            bg = np.zeros((w,w))
            for y in range(h):
                for x in range(w):
                    dif = floor((w - h)/2)
                    bg[y + dif,x] = image[y,x]
        else:
            bg = image
        img = cv2.resize(image,(24,24), interpolation = cv2.INTER_LINEAR)
        bg = np.zeros((28, 28))
        for y in range(24):
            for x in range(24):
                bg[y + 2,x + 2] = img[y,x]
        return bg
    def resize_to_train(self,image):
        h,w = image.shape
        blur = cv2.GaussianBlur(image,(5,5),0)
        ret3,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = (255 - image)
        bg = None
        if(h > w):
            bg = np.zeros((h,h))
            for y in range(h):
                for x in range(w):
                    dif = floor((h - w)/2)
                    bg[y,x + dif] = image[y,x]
        elif(w > h):
            bg = np.zeros((w,w))
            for y in range(h):
                for x in range(w):
                    dif = floor((w - h)/2)
                    bg[y + dif,x] = image[y,x]
        else:
            bg = image
        return cv2.resize(bg,(28,28),interpolation = cv2.INTER_LINEAR)

    def resize_28_28(self,image):
        h,w = image.shape
        if(w != h):
            if(w < 28):
                #cal the min and the max
                xmin = float('inf')
                xmax = float('-inf')
                for y in range(h):
                    for x in range(w):
                        if(image[y,x] == 255):
                            xmin = min(x,xmin)
                            xmax = max(x,xmax)
                image = cv2.resize(image,(w,28), interpolation = cv2.INTER_LINEAR)
                bg = np.zeros((28, 28))
                res = 28 - (xmax - xmin)
                res_half_1 = floor(res/2)
                res_half_2 = res - res_half_1
                for y in range(28):
                    for x in range(w):
                        bg[y,x + res_half_1] = image[y,x]
                image = bg
        else:
            image = cv2.resize(image,(28,28), interpolation = cv2.INTER_LINEAR)
        return image
"""pp = Preproc()
cv2.imshow("imagen",pp.resize_to_train(cv2.imread("data/printable/73/a01-014-02-07.png",0)))
cv2.waitKey(0);"""
