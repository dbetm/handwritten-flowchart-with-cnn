from mnist import MNIST
import cv2
import numpy as np
import os
from random import randint
from data_preparation import Preproc
class DataGenerator(object):
    def __init__(self,eminist_path):
        self.EMNIST_PATH = eminist_path
        self.PRINTABLE = "data/printable"
        self.get_data()
    def get_data(self):
        emnist = MNIST(self.EMNIST_PATH)
        emnist.select_emnist('byclass')
        #get the images and the labels of the mninst for training and testing
        emnist_images_train,emnist_labels_train = emnist.load_training()
        emnist_images_test,emnist_labels_test = emnist.load_testing()
        emnist_images_train = np.array(emnist_images_train[0:500000])
        emnist_images_test = np.array(emnist_images_test)
        emnist_images_train = emnist_images_train.reshape(emnist_images_train.shape[0],28,28,1)
        emnist_images_test = emnist_images_test.reshape(emnist_images_test.shape[0],28,28,1)

        emnist_images_train = emnist_images_train.astype('uint8')
        emnist_images_test = emnist_images_test.astype('uint8')
        #Get the printable data for append to the train set
        printable_images_train = []
        printable_labels_train = []
        printable_images_test = []
        printable_labels_test = []
        pp = Preproc()

        printable_set = os.listdir(self.PRINTABLE)
        for dir in printable_set:
            images = os.listdir(self.PRINTABLE+"/"+str(dir))
            num_split = int(len(images) * 80 / 100)
            for image in range(num_split):
                pos = randint(0,len(images)-1)
                printable_images_train.append(pp.resize_to_train(cv2.imread(self.PRINTABLE+"/"+str(dir)+"/"+str(images[pos]),0)))
                printable_labels_train.append(int(dir))
                images.pop(pos)
            for img in images:
                printable_images_test.append(pp.resize_to_train(cv2.imread(self.PRINTABLE+"/"+str(dir)+"/"+str(img),0)))
                printable_labels_test.append(int(dir))
        printable_images_train = np.array(printable_images_train)
        printable_images_test = np.array(printable_images_test)
        printable_labels_train = np.array(printable_labels_train)
        printable_labels_test = np.array(printable_labels_test)
        printable_images_train = printable_images_train.reshape(printable_images_train.shape[0],28,28,1)
        printable_images_test = printable_images_test.reshape(printable_images_test.shape[0],28,28,1)
        print("Printable",printable_images_train.shape,printable_images_test.shape,printable_labels_train.shape,printable_labels_test.shape)
        print("Emint",emnist_images_train.shape,emnist_images_test.shape)
        images_train = np.concatenate((printable_images_train,emnist_images_train),axis = 0)
        images_test = np.concatenate((printable_images_test,emnist_images_test),axis = 0)
        labels_train = np.concatenate((printable_labels_train,emnist_labels_train[0:500000]),axis = 0)
        labels_test = np.concatenate((printable_labels_test,emnist_labels_test),axis = 0)
        print("despues",images_train.shape,images_test.shape,labels_train.shape,labels_test.shape)


        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("shape",images_train.shape,images_test.shape)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        return images_train , labels_train , images_test , labels_test
