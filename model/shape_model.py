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

import os

from train import Trainer

class ShapeModel(object):
    """ShapeModel allows to start and manager the training and test process
    in a deep learning arquitecture for object detection, such is Faster R-CNN.
    The implementation is based in code from: https://github.com/kbardool/keras-frcnn
    Mirror link: https://github.com/dbetm/keras-frcnn
    """

    def __init__(self, dataset_path, num_rois=32, weights_input_path="none"):

        super(ShapeModel, self).__init__()

        self.dataset_path = dataset_path
        self.num_rois = 32
        self.weights_input_path = weights_input_path

    def __generate_results_path(self, base):

        ans = base + "_results"
        folder = os.listdir(ans)
        num_results = len(folder)

        return ans + "/" + str(num_results)

    def train(
            self,
            horizontal_flips,
            vertical_flips,
            num_epochs=5,
            epoch_length=32,
            learning_rate=1e-5,
            num_rois=32,
            use_gpu=False,
    ):
        """Fit deep learning model."""

        # Initialize paths when creating the results folder
        base_path = self.__generate_results_path("training")
        annotate_path = base_path + "/annotate.txt"
        weights_output_path = base_path + "/flowchart_3b_model.hdf5"
        config_output_filename = base_path + "/config.pickle"
        # Create folder training folder
        os.mkdir(base_path)
        # Instance Trainer
        trainer = Trainer(use_gpu)
        # Recover data from dataset
        trainer.recover_data(
            self.dataset_path,
            annotate_path,
            generate_annotate=True
        )
        # Configure trainer
        trainer.configure(
            horizontal_flips,
            vertical_flips,
            self.num_rois,
            weights_output_path,
            self.weights_input_path,
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            learning_rate=learning_rate,
        )
        trainer.save_config(config_output_filename)
        exit() # DEBUG
        trainer.train()


if __name__ == '__main__':
    shape_model = ShapeModel(
        dataset_path="/home/david/Escritorio/flowchart-3b(splitter)",
        num_rois=32
    )
    # testing train
    shape_model.train(
        horizontal_flips=False,
        vertical_flips=False,
        num_epochs=1,
    )
