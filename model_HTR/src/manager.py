import keras_ocr
import tensorflow as tf
from htr import HTR
import string
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)
class Manager(object):
    def __init__(self):
        self.alphabet = string.printable[:36]
        self.recognizer = keras_ocr.recognition.Recognizer(alphabet = self.alphabet)
        self.recognizer.compile()
        self.pipeline = keras_ocr.pipeline.Pipeline(recognizer = self.recognizer)
        self.text_recognizer = HTR()
        self.image = None
    def detect(self,image_path):
        images = keras_ocr.tools.read(image_path)
        self.image = images
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
    def recognize(self,image_path):
        boxes = self.detect(image_path)
        for box in boxes:
            y2,y1,x2,x1 = box
            crop_img = self.image[y1:y2, x1:x2]
            print(self.text_recognizer.predict(crop_img))
m = Manager()
m.recognize("../../Images/diagrama.jpg")
