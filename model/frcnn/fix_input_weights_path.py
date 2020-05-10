# -*- coding: utf-8 -*-

""" Copy this file inside the folder of the results and exec...
"""

import pickle

if __name__ == '__main__':
    # open config file
    config_path = input("Please, type relative path of the config file (.pickle):\n")
    with open(config_path, 'rb') as f_in:
        config = pickle.load(f_in)

    new_path = input("Please, type path (weights) taking as base folder 'model/':\n")

    old_path = config.weights_output_path
    config.weights_output_path = new_path

    # rewrite file
    with open(config_path, 'wb') as config_f:
        pickle.dump(config, config_f)

    print("Fix path, old: {}, new: {}".format(old_path, new_path))
