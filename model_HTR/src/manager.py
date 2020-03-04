import keras_ocr
import tensorflow as tf
from htr import HTR
class Manager(object):
    def __init__():
        self.alphabet = string.printable[:36]
        self.recognizer = keras_ocr.recognition.Recognizer(alphabet = self.alphabet)
        self.recognizer.compile()
        self.pipeline = keras_ocr.pipeline.Pipeline(recognizer = recognizer)
        self.text_recognizer = HTR()
    def detect(self,image_path):
        images = keras_ocr.tools.read(self.image_path)
        prediction_groups = self.pipeline.recognize([images])
        results = []
        for ibox in prediction_groups[0]:
            box = ibox[1]
            xs,ys = set(),set()
            for x in box:
                xs.add(x[0])
                ys.add(x[1])
            results.append(list(map(int,[max(ys),min(ys),max(xs),min(xs)])))
        return results
    def recognize(self):
        boxes = self.detect_text()
        for box in boxes:
            y2,y1,x2,x1 = box
            crop_img = self.image[y1:y2, x1:x2]
            self.text_recognizer.predict(crop_img)
