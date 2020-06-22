"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""

from itertools import groupby
from . import preproc as pp
import h5py
import numpy as np
import unicodedata
import cv2
import os
source = "iam"
new = "new_data"
new_source_path = os.path.join("text_model","data_model",f"{new}.hdf5")
source_path = os.path.join("text_model","data_model",f"{source}.hdf5")
class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, source, batch_size, charset, max_text_length, predict=False,load_data = True):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.batch_size = batch_size
        self.partitions = ['test'] if predict else ['train', 'valid', 'test']
        self.size = dict()
        self.steps = dict()
        self.index = dict()
        self.dataset = dict()
        self.new_dataset = dict()

        if load_data:
            with h5py.File(source, "r") as f:
                for pt in self.partition:
                    self.dataset[pt] = dict()

                    self.dataset[pt]['dt'] = np.array(f[pt]['dt'])
                    self.dataset[pt]['gt'] = np.array(f[pt]['gt'])

                    randomize = np.arange(len(self.dataset[pt]['gt']))
                    np.random.seed(42)
                    np.random.shuffle(randomize)

                    self.dataset[pt]['dt'] = self.dataset[pt]['dt'][randomize]
                    self.dataset[pt]['gt'] = self.dataset[pt]['gt'][randomize]
            for pt in self.partitions:
                # decode sentences from byte
                self.dataset[pt]['gt'] = [x.decode() for x in self.dataset[pt]['gt']]

                # set size and setps
                self.size[pt] = len(self.dataset[pt]['gt'])
                self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))
                self.index[pt] = 0
    def load_data(self):
        with h5py.File(source_path, "r") as f:
            for pt in ['valid']:
                self.dataset[pt] = dict()

                self.dataset[pt]['dt'] = np.array(f[pt]['dt'])
                self.dataset[pt]['gt'] = np.array(f[pt]['gt'])

                randomize = np.arange(len(self.dataset[pt]['gt']))
                np.random.seed(42)
                np.random.shuffle(randomize)

                self.dataset[pt]['dt'] = self.dataset[pt]['dt'][randomize]
                self.dataset[pt]['gt'] = self.dataset[pt]['gt'][randomize]
        for pt in ['valid']:
            # decode sentences from byte
            self.dataset[pt]['gt'] = [x.decode() for x in self.dataset[pt]['gt']]

            # set size and setps
            self.size[pt] = len(self.dataset[pt]['gt'])
            self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))
            self.index[pt] = 0
        with h5py.File(new_source_path, "r") as f:
                self.new_dataset['dt'] = np.array(f['dt'])
                self.new_dataset['gt'] = np.array(f['gt'])

                randomize = np.arange(len(self.new_dataset['gt']))
                np.random.seed(42)
                np.random.shuffle(randomize)

                self.new_dataset['dt'] = self.new_dataset['dt'][randomize]
                self.new_dataset['gt'] = self.new_dataset['gt'][randomize]
        # decode sentences from byte
        self.new_dataset['gt'] = [x.decode() for x in self.new_dataset['gt']]

        # set size and setps
        self.size['train'] = len(self.new_dataset['gt'])
        self.steps['train'] = int(np.ceil(self.size['train'] / self.batch_size))
        self.index['train'] = 0
    def new_next_train_batch(self):
        """Get the next batch from train partition (yield)"""
        while True:
            if self.index['train'] >= self.size['train']:
                self.index['train'] = 0

            index = self.index['train']
            until = self.index['train'] + self.batch_size
            self.index['train'] = until

            x_train = self.new_dataset['dt'][index:until]
            x_train = pp.augmentation(x_train,
                                      rotation_range=1.5,
                                      scale_range=0.05,
                                      height_shift_range=0.025,
                                      width_shift_range=0.05,
                                      erode_range=5,
                                      dilate_range=3)
            x_train = pp.normalization(x_train)

            y_train = [self.tokenizer.encode(y) for y in self.new_dataset['gt'][index:until]]
            y_train = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_train]
            y_train = np.asarray(y_train, dtype=np.int16)

            yield (x_train, y_train)
    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""
        while True:
            if self.index['train'] >= self.size['train']:
                self.index['train'] = 0

            index = self.index['train']
            until = self.index['train'] + self.batch_size
            self.index['train'] = until

            x_train = self.dataset['train']['dt'][index:until]
            x_train = pp.augmentation(x_train,
                                      rotation_range=1.5,
                                      scale_range=0.05,
                                      height_shift_range=0.025,
                                      width_shift_range=0.05,
                                      erode_range=5,
                                      dilate_range=3)
            x_train = pp.normalization(x_train)

            y_train = [self.tokenizer.encode(y) for y in self.dataset['train']['gt'][index:until]]
            y_train = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_train]
            y_train = np.asarray(y_train, dtype=np.int16)

            yield (x_train, y_train)

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        while True:
            if self.index['valid'] >= self.size['valid']:
                self.index['valid'] = 0

            index = self.index['valid']
            until = self.index['valid'] + self.batch_size
            self.index['valid'] = until

            x_valid = self.dataset['valid']['dt'][index:until]
            x_valid = pp.normalization(x_valid)

            y_valid = [self.tokenizer.encode(y) for y in self.dataset['valid']['gt'][index:until]]
            y_valid = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_valid]
            y_valid = np.asarray(y_valid, dtype=np.int16)

            yield (x_valid, y_valid)

    def next_test_batch(self):
        """Return model predict parameters"""

        while True:
            if self.index['test'] >= self.size['test']:
                self.index['test'] = 0
                break

            index = self.index['test']
            until = self.index['test'] + self.batch_size
            self.index['test'] = until

            x_test = self.dataset['test']['dt'][index:until]
            x_test = pp.normalization(x_test)
            yield x_test



class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())

        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        decoded = pp.text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
