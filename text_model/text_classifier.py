import os
import datetime
import string
import numpy as np
import keras_ocr
from math import ceil
from . data.generator import DataGenerator
from . network.model import HTRModel
from . data import preproc as pp
import cv2
from node import Node

alphabet_recognition = string.printable[:95]
alphabet_detection = string.printable[:36]
input_size = (1024,128,1)
max_text_length = 128
batch_size = 16
source = "iam"
arch = "puigcerver"
source_path = os.path.join("text_model","data_model",f"{source}.hdf5")
output_path = os.path.join("text_model","output",source,arch)
target_path = os.path.join(output_path,"checkpoint_weights.hdf5")

from keras import backend as K
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
tf.config.experimental.list_physical_devices(device_type=None)
class TextClassifier(object):
    def __init__(self):
        """
        All the values that need initialize are added
        """
        self.dtgen = DataGenerator(source = source_path,batch_size = batch_size,charset = alphabet_recognition,max_text_length = max_text_length,load_data = False)
        self.model = HTRModel(architecture = arch,input_size = input_size,vocab_size = self.dtgen.tokenizer.vocab_size)
        self.model.load_checkpoint(target = target_path)

        self.recognizer = keras_ocr.recognition.Recognizer(alphabet = alphabet_detection)
        self.recognizer.compile()
        self.pipeline = keras_ocr.pipeline.Pipeline(recognizer = self.recognizer)

    def train_new_data(self):
        self.dtgen.load_data()
        callbacks = self.model.get_callbacks_continue(logdir=output_path, checkpoint=target_path, verbose=1)
        self.model.fit(x=self.dtgen.new_next_train_batch(),
              epochs=1,
              steps_per_epoch=self.dtgen.steps['train'],
              validation_data=self.dtgen.next_valid_batch(),
              validation_steps=self.dtgen.steps['valid'],
              callbacks=callbacks,
              shuffle=True,
              verbose=1)
    def __set_image(self,image_path):
        image = cv2.imread(image_path,0)
        blur = cv2.GaussianBlur(image,(5,5),0)
        ret3,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image
    def __exist_character_x(self,image,x,ymin,ymax):
        for y in range(ymin,ymax + 1):
            if(image[y,x] == 0):
                return True
        return False
    def __exist_character_y(self,image,y,xmin,xmax):
        for x in range(xmin,xmax + 1):
            if(image[y,x] == 0):
                return True
        return False
    def __is_collapse(self,A,B):
        """ This function return if exist a interseccion in two nodes.
        """
        coordinateA = A
        coordinateB = B
        x = max(coordinateA[0], coordinateB[0])
        y = max(coordinateA[2], coordinateB[2])
        w = min(coordinateA[1], coordinateB[1]) - x
        h = min(coordinateA[3], coordinateB[3]) - y
        if w < 0 or h < 0:
        	return False
        return True
    def __merge_text_nodes(self,image_path,text_coord):
        global text_nodes
        text_nodes = text_coord
        image = self.__set_image(image_path)
        EXPAND_MAX = int(image.shape[1] / 20)
        to_delate = []
        text_node_up = text_nodes
        for i in text_nodes:
            if not (i in to_delate):
                while True:
                    xmin,xmax,ymin,ymax = i
                    collapse_times = 0
                    for j in text_nodes:
                        if not (j in to_delate):
                            if(i != j):
                                if(self.__is_collapse([xmin - EXPAND_MAX,xmax + EXPAND_MAX,ymin,ymax],j)):
                                    xmin_A,xmax_A,ymin_A,ymax_A = i
                                    xmin_B,xmax_B,ymin_B,ymax_B = j
                                    n_xmin = min(xmin_A,xmin_B)
                                    n_xmax = max(xmax_A,xmax_B)
                                    n_ymin = min(ymin_A,ymin_B)
                                    n_ymax = max(ymax_A,ymax_B)
                                    text_nodes[text_nodes.index(i)] = [n_xmin,n_xmax,n_ymin,n_ymax]
                                    i = [n_xmin,n_xmax,n_ymin,n_ymax]
                                    to_delate.append(j)
                                    collapse_times += 1
                    if(collapse_times == 0):
                        break
        if len(to_delate) > 0:
            for x in to_delate:
                text_nodes.remove(x)
        EXPAND_MAX = 10
        res = []
        for i in text_nodes:
            xmin,xmax,ymin,ymax = i
            """expand_max = EXPAND_MAX

            while(True):
                xmin = xmin - 1
                expand_max -= 1
                if(expand_max == 0):
                    break
                query = self.__exist_character_x(image,xmin,ymin,ymax)
                if(query == True):
                    while(self.__exist_character_x(image,xmin,ymin,ymax)):
                        xmin = xmin - 1
                    break
            #i = [xmin,xmax,ymin,ymax]
            #xmin,xmax,ymin,ymax = i

            expand_max = EXPAND_MAX
            while(True):
                xmax = xmax + 1
                expand_max -= 1
                if(expand_max == 0):
                    break
                query = self.__exist_character_x(image,xmax,ymin,ymax)
                if(query == True):
                    while(self.__exist_character_x(image,xmax,ymin,ymax)):
                        xmax = xmax + 1
                    break
            #i = [xmin,xmax,ymin,ymax]
            #xmin,xmax,ymin,ymax = i
            expand_max = EXPAND_MAX
            while(True):
                ymin = ymin - 1
                expand_max -= 1
                if(expand_max == 0):
                    break
                query = self.__exist_character_y(image,ymin,xmin,xmax)
                if(query == True):
                    while(self.__exist_character_y(image,ymin,xmin,xmax)):
                        ymin = ymin - 1
                    break

            #i = [xmin,xmax,ymin,ymax]
            #xmin,xmax,ymin,ymax = i
            expand_max = EXPAND_MAX
            while(True):
                ymax = ymax + 1
                expand_max -= 1
                if(expand_max == 0):
                    break
                query = self.__exist_character_y(image,ymax,xmin,xmax)
                print(query)
                if(query == True):
                    while(self.__exist_character_y(image,ymax,xmin,xmax)):
                        ymax = ymax + 1
                    break"""

            res.append([xmin - EXPAND_MAX,xmax + EXPAND_MAX,ymin - int(EXPAND_MAX//2),ymax + int(EXPAND_MAX//2)])
        return res
    def __get_bbox(self,image_path):
        images = keras_ocr.tools.read(image_path)
        self.image = images
        prediction_groups = self.pipeline.recognize([images])
        texts = []
        results = []
        for ibox in prediction_groups[0]:
            box = ibox[1]
            texts.append(ibox[0])
            xs,ys = set(),set()
            for x in box:
                xs.add(x[0])
                ys.add(x[1])
            results.append(list(map(ceil,[max(ys),min(ys),max(xs),min(xs)])))
        return results,texts
    def draw_boxes(self,image_path,boxes):
        image = cv2.imread(image_path)
        for box in boxes:
            xmin,xmax,ymin,ymax = box
            pts = np.array([[xmin,ymin], [xmax,ymin],[xmax,ymax],[xmin,ymax]], np.int32)
            cv2.polylines(img=image,pts=np.int32([pts]),color=(255, 0, 0),thickness=5,isClosed=True)
        cv2.imshow("image",cv2.resize(image,(0, 0),fx = 0.3, fy = 0.3))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def __image_generator(self,image):
        x_predict = pp.resize_new_data(image,input_size[:2])
        x_predict = np.array([x_predict])
        x_predict = pp.normalization(x_predict)
        yield x_predict
    def recognize(self,image_path):
        boxes,texts = self.__get_bbox(image_path)
        images = []
        coords = []
        txts = []
        nodes = []
        for box,text in zip(boxes,texts):
            y2,y1,x2,x1 = box
            coords.append([x1,x2,y1,y2])
        coords = self.__merge_text_nodes(image_path,coords)
        self.draw_boxes(image_path,coords)
        for box in coords:
            xmin,xmax,ymin,ymax = box
            crop_img = self.image[ymin:ymax, xmin:xmax]
            images.append(crop_img)
            predict,prob = self.model.predict(x = self.__image_generator(crop_img),steps = 1,ctc_decode = True , verbose = 1)
            predict = [self.dtgen.tokenizer.decode(x[0]) for x in predict]
            text = predict[0]
            nodes.append(Node(coordinate=[xmin,xmax,ymin,ymax],text = text))
        return zip(nodes,images)
