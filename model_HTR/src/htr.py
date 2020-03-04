import os
import datetime
import string
from data.generator import DataGenerator
from data import preproc as pp
from network.model import HTRModel
import cv2

class HTR(object):
    def __init__(self):
        self.source = "iam"
        self.arch = "flor"
        self.batch_size = 16
        self.source_path = os.path.join("..", "data", f"{self.source}.hdf5")
        self.output_path = os.path.join("..", "output", self.source, self.arch)
        self.target_path = os.path.join(self.output_path, "checkpoint_weights.hdf5")
        self.input_size = (1024, 128, 1)
        self.max_text_length = 128
        self.charset_base = string.printable[:95]

        self.dtgen = DataGenerator(source=self.source_path,
                              batch_size=self.batch_size,
                              charset=self.charset_base,
                              max_text_length=self.max_text_length)
        self.model = HTRModel(architecture=self.arch, input_size=self.input_size, vocab_size=self.dtgen.tokenizer.vocab_size)
        self.model.load_checkpoint(target=self.target_path)
    def predict(self,image):
        predicts, _ = self.model.predict(x=self.dtgen.recognize_image(image),steps=1,ctc_decode=True,verbose=1)
        predicts = [self.dtgen.tokenizer.decode(x[0]) for x in predicts]
        return predicts
