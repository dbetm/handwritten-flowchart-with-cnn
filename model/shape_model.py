# -*- coding: utf-8 -*-

"""
Shape model module contains a class to manage the model for shapes and connectors
recognition, allows train an architecture that use CNNs.
"""

__autor__ = "David"
__credits__ = ["David Betancourt Montellano", "Onder Francisco Campos García"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "davbetm@gmail.com"
__status__ = "Development"


class ShapeModel(object):
    """ShapeModel allows to start and manager the training and test process
    in a deep learning arquitecture for object detection, such is Faster R-CNN.
    The implementation is based in code from: https://github.com/kbardool/keras-frcnn
    Mirror link: https://github.com/dbetm/keras-frcnn
    """

    def __init__(
            self,
            batch_size,
            learning_rate,
            train_data_path,
            test_data_path,
            epochs):

        super(ShapeModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.channels = 1 # Number of color channels
        self.history = None # Records about training process
        self.num_classes = None # Assign after data parser
        self.epochs = epochs
        self.num_train_samples = None # Assign after data parser

    def get_batch_size(self):
        return self.batch_size

    def get_num_classes(self):
        return self.num_classes

    def get_learning_rate(self):
        return self.learning_rate

    def get_num_channels(self):
        return self.num_channels

    def get_num_train_samples(self):
        return self.num_train_samples

    def load_dataset(self):
        """ Do data parsing and split in training and validation dataset """
        pass

    def train(self):
        """ Run evolutive process """
        pass


if __name__ == '__main__':
    frcnn0 = ShapeModel(32, 0.0001, "train_images/", "test_images/", 500)
    print(frcnn0.get_batch_size())
