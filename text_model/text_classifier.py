import os
import datetime
import string
import numpy as np
from data.generator import DataGenerator
from network.model import HTRModel
from data import preproc as pp
import cv2
alphabet_recognition = string.printable[:95]
alphabet_detection = string.printable[:36]
input_size = (1024,128,1)
max_text_length = 128
batch_size = 16
source = "iam"
arch = "puigcerver"
source_path = os.path.join("data_model",f"{source}.hdf5")
output_path = os.path.join("output",source,arch)
target_path = os.path.join(output_path,"checkpoint_weights.hdf5")
class TextClassifier(object):
    def __init__(self):
        """
        All the values that need initialize are added
        """
        self.dtgen = DataGenerator(source = source_path,batch_size = batch_size,charset = alphabet_recognition,max_text_length = max_text_length)
        self.model = HTRModel(architecture = arch,input_size = input_size,vocab_size = self.dtgen.tokenizer.vocab_size)
        self.model.load_checkpoint(target = target_path)
        """
        self.recognizer = keras_ocr.recognition.Recognizer(alphabet = alphabet_detection)
        self.recognizer.compile()
        self.pipeline = keras_ocr.pipeline.Pipeline(recognizer = self.recognizer)"""
        #self.text_model = EH_model()
    """def __get_bbox(self,image_path):
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
        return results,texts"""
    def __image_generator(self,image):
        x_predict = pp.resize_new_data(image,input_size[:2])
        x_predict = np.array([x_predict])
        x_predict = pp.normalization(x_predict)
        yield x_predict
    def recognize(self,image_path):
        """boxes,texts = self.__get_bbox(image_path)
        nodes = []
        for box,text in zip(boxes,texts):
            y2,y1,x2,x1 = box
            crop_img = self.image[y1:y2, x1:x2]"""
        predict,prob = self.model.predict(x = self.__image_generator(cv2.imread(image_path)),steps = 1,ctc_decode = True , verbose = 1)
        predict = [self.dtgen.tokenizer.decode(x[0]) for x in predict]
        text = predict[0]
        print(text)
        #nodes.append(Node(coordinate=[x1,x2,y1,y2],text = text))
        #return nodes
tc = TextClassifier()
tc.recognize("images/hola2.png")
