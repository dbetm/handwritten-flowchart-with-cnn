import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import keras_ocr
import tensorflow as tf
class Segementer(object):
    def __init__(self,image_path,trained_model_path):
        self.image = cv2.imread(image_path)
        self.image_path = image_path
        self.trained_model_path = trained_model_path
        #add this line in the main to load the model in the init of the proyect not for every deteccion
        self.pipeline = keras_ocr.pipeline.Pipeline()
    def detect_text(self):
        images = keras_ocr.tools.read(self.image_path)
        prediction_groups = self.pipeline.recognize([images])
        print("Tamaño",len(prediction_groups[0]))
        results = []
        for ibox in prediction_groups[0]:
            box = ibox[1]
            xs,ys = set(),set()
            for x in box:
                xs.add(x[0])
                ys.add(x[1])
            result = frozenset()
            result = result|xs
            result = result|ys
            result = list(result)
            result = list(map(int,result))
            results.append(result)
        return results
    def segment(self):
        boxes = self.detect_text()
        print(boxes)
        images = []
        for box in boxes:
            y1,y2,x1,x2 = box
            crop_img = self.image[y1:y2, x1:x2]
            images.append(crop_img)
        print(images)
seg = Segementer("Images/diagrama.jpg","path")
seg.segment()
