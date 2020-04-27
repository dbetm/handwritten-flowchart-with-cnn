from EMNIST_ HASY_model import model.Model as EH_model
import keras_ocr
import string
sys.path.append('..')
from node import Node
class TexClassifier(object):
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
        self.text_model = EH_model()
    def __get_bbox(self):
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
        boxes = self.__get_bbox(image_path)
        nodes = []
        for box in boxes:
            y2,y1,x2,x1 = box
            crop_img = self.image[y1:y2, x1:x2]
            text_predict = self.text_model.separete_characteres(crop_img)
            nodes.append(Node(coordinate=[x1,x2,y1,y2],text = text_predict))
        return nodes
