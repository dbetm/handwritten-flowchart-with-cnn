"""Dataset reader and process"""

import os
import html
import string
import xml.etree.ElementTree as ET

from . import preproc as pp
from functools import partial
from glob import glob
from multiprocessing import Pool
import h5py
new = "new_data"
source = "iam"
arch = "puigcerver"
new_source_path = os.path.join("text_model","data_model",f"{new}.hdf5")
source_path = os.path.join("text_model","data_model",f"{source}.hdf5")
output_path = os.path.join("text_model","output",source,arch)
target_path = os.path.join(output_path,"checkpoint_weights.hdf5")
class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""
        self.dataset = getattr(self, f"_{self.name}")()


    def preprocess_partitions(self, input_size):
        """Preprocess images and sentences from partitions"""

        for y in self.partitions:
            arange = range(len(self.dataset[y]['gt']))

            for i in reversed(arange):
                text = pp.text_standardize(self.dataset[y]['gt'][i])

                if not self.check_text(text):
                    self.dataset[y]['gt'].pop(i)
                    self.dataset[y]['dt'].pop(i)
                    continue

                self.dataset[y]['gt'][i] = text.encode()

            pool = Pool()
            self.dataset[y]['dt'] = pool.map(partial(pp.preprocess, input_size=input_size), self.dataset[y]['dt'])
            pool.close()
            pool.join()


    def _iam(self):
        """IAM dataset reader"""
        pt_path = os.path.join(self.source, "largeWriterIndependentTextLineRecognitionTask")
        paths = {"train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "validationset1.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splitted = line.split()

            if splitted[1] == "ok":
                gt_dict[splitted[0]] = " ".join(splitted[8::]).replace("|", " ")

        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                try:
                    split = line.split("-")
                    folder = f"{split[0]}-{split[1]}"

                    img_file = f"{split[0]}-{split[1]}-{split[2]}.png"
                    img_path = os.path.join(self.source, "lines", split[0], folder, img_file)

                    dataset[i]['gt'].append(gt_dict[line])
                    dataset[i]['dt'].append(img_path)
                except KeyError:
                    pass

        return dataset
    def add_new_val(self,images,words):
        dataset = dict()
        dataset["train"] = {"dt": [], "gt": []}
        dataset["valid"] = {"dt": [], "gt": []}
        dataset["test"] = {"dt": [], "gt": []}
        for i in data:
            text = pp.text_standardize(i[1])
            if self.check_text(text):
                dataset["valid"]['gt'].append(text.encode())
            #change the preprocess
            dataset["valid"]["dt"].append(pp.preprocess(i[0],input_size))
        hf = h5py.File(target_pat,'a')
        for i in self.partitions:
            hf.create_dataset(f"{i}/dt", data=dataset[i]['dt'], compression="gzip", compression_opts=9)
            hf.create_dataset(f"{i}/gt", data=dataset[i]['gt'], compression="gzip", compression_opts=9)
        hf.close()
    def save_new_data(self,images,words):
        dataset = dict()
        dataset = dict()
        dataset = {"dt": [], "gt": []}
        for image,word in zip(images,words):
            text = pp.text_standardize(word)
            if self.check_text(text):
                dataset['gt'].append(text.encode())
                #change the preprocess
                dataset["dt"].append(pp.resize_new_data(image,(1024,128)))
        if(os.path.isfile(new_source_path)):
            os.remove(new_source_path)
        hf = h5py.File(new_source_path,'a')
        hf.create_dataset("/dt", data=dataset['dt'], compression="gzip", compression_opts=9)
        hf.create_dataset("/gt", data=dataset['gt'], compression="gzip", compression_opts=9)
        hf.close()
        dataset.clear()
    @staticmethod
    def check_text(text):
        """Make sure text has more characters instead of punctuation marks"""

        strip_punc = text.strip(string.punctuation).strip()
        no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

        if len(text) == 0 or len(strip_punc) == 0 or len(no_punc) == 0:
            return False

        punc_percent = (len(strip_punc) - len(no_punc)) / len(strip_punc)

        return len(no_punc) >= 2 and punc_percent <= 0.1
