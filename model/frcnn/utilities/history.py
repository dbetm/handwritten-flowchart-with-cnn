# -*- coding: utf-8 -*-
from keras.utils.vis_utils import plot_model
import json

class History(object):
    """The results of the training persist, as well as its hyperparameters."""

    def __init__(self, base_path):
        super(History, self).__init__()
        self.base_path = base_path
        # Create CSV log
        self.csv_log = open(base_path + "/history.csv", "x")
        self.csv_log.write(
            "epoch_num," +
            "mean_overlapping_bboxes," +
            "class_acc," +
            "loss," +
            "loss_rpn_cls," +
            "loss_rpn_regr," +
            "loss_det_cls," +
            "loss_det_regr," +
            "time"
        )
        self.csv_log.close()

    def save_summary(self, model, name):
        """Save summary of models in txt file."""

        path = self.base_path + "/summary_model_" + name + ".txt"
        file = open(path, "x")
        model.summary(print_fn=lambda x: file.write(x + '\n'))
        file.close()

    def save_model_image(self, model, name):
        """Save image of the models like png file."""

        path = self.base_path + "/plot_model_" + name + ".png"
        plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)

    def save_classes_info(self, classes):
        num_classes = len(classes)
        path = self.base_path + "/classes_info.txt"
        file = open(path, "x")
        file.write("Training images per class:\n")
        info = json.dumps(classes)
        file.write(info)
        file.write("\nNum classes (including bg):\n")
        file.write(str(num_classes))
        file.close()

    def save_best_model(self, model, path):
        """Save weights of the best model."""

        model.save_weights(path)

    def append_epoch_info(self, row):
        """Open existing log cvs file and append a new row epoch info."""

        self.csv_log = open(self.base_path + "/history.csv", "a")
        string = ','.join(str(s) for s in row)
        self.csv_log.write("\n" + string)
        self.csv_log.close()
