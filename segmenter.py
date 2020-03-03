import matplotlib.pyplot as plt
import cv2
import keras_ocr
import tensorflow as tf
import string
class Segementer(object):
    def __init__(self,image_path,trained_model_path):
        self.image = cv2.imread(image_path)
        self.image_path = image_path
        self.trained_model_path = trained_model_path
        #add this line in the main to load the model in the init of the proyect not for every deteccion
        #keras_ocr.data_generation.get_fonts(alphabet=alphabet)
        alphabet = string.printable[:36]
        #print(alphabet)
        recognizer = keras_ocr.recognition.Recognizer(alphabet = alphabet)
        #recognizer.compile()
        #recognizer.training_model.fit_generator(generator=keras_ocr.data_generation.get_fonts(alphabet=alphabet),steps_per_epoch=100,epochs=100)
        self.pipeline = keras_ocr.pipeline.Pipeline(recognizer = recognizer)

    def detect_text(self):
        images = keras_ocr.tools.read(self.image_path)
        prediction_groups = self.pipeline.recognize([images])
        print("Tamaño",len(prediction_groups[0]))
        results = []
        for ibox in prediction_groups[0]:
            print(ibox[0])
            box = ibox[1]
            xs,ys = set(),set()
            for x in box:
                xs.add(x[0])
                ys.add(x[1])
            results.append(list(map(int,[max(ys),min(ys),max(xs),min(xs)])))
        return results
    def segment(self):
        boxes = self.detect_text()
        print(boxes)
        images = []
        for box in boxes:
            y2,y1,x2,x1 = box
            crop_img = self.image[y1:y2, x1:x2]
            plt.imshow(crop_img)
            plt.show()
seg = Segementer("Images/diagrama.jpg","path")
seg.segment()
