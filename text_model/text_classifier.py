import keras_ocr
import string
import os
import sys
from . EHmodel.model import Model as EH_model
from node import Node
from math import ceil
class TextClassifier(object):
    def __init__(self):
        """
        All the values that need initialize are added
        """
        #Keras-ocr library is add to text detect
        self.alphabet = string.printable[:36]
        self.recognizer = keras_ocr.recognition.Recognizer(alphabet = self.alphabet)
        self.recognizer.compile()
        self.pipeline = keras_ocr.pipeline.Pipeline(recognizer = self.recognizer)
        #EMINIST + HASY model is added to text classifier
        #self.text_model = EH_model()
    def __get_bbox(self,image_path):
        images = keras_ocr.tools.read(image_path)
        self.image = images
        prediction_groups = self.pipeline.recognize([images])
        texts = []
        results = []
        for ibox in prediction_groups[0]:
            box = ibox[1]
            print("--------box-----------",box)
            texts.append(ibox[0])
            xs,ys = set(),set()
            for x in box:
                xs.add(x[0])
                ys.add(x[1])
            results.append(list(map(ceil,[max(ys),min(ys),max(xs),min(xs)])))
        return results,texts
    def recognize(self,image_path):
        boxes,texts = self.__get_bbox(image_path)
        nodes = []
        for box,text in zip(boxes,texts):
            y2,y1,x2,x1 = box
            crop_img = self.image[y1:y2, x1:x2]
            #text_predict = self.text_model.separete_characteres(crop_img)
            print("-----------bbox despues------------",x1,x2,y1,y2)
            nodes.append(Node(coordinate=[x1,x2,y1,y2],text = text))
        return nodes
