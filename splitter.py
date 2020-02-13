import os
import shutil
import argparse
from random import randint
dataset_path = "flowchart-3b"
training_set_path = "data/train"
validation_set_path = "data/validation"
split_tam = 80
def move_archive(list,path,class_i,type):
    global split_tam
    num_files = len(list)
    copy_path = "images" if type==0 else "annots"
    #Calculate the number of files that should be moved to the training set
    num_split = int(num_files * split_tam / 100)
    for i in range(num_split):
        pos = randint(0,len(list)-1)
        shutil.copy(path+str(list[pos]),training_set_path+"/"+class_i+"/"+copy_path)
        list.pop(pos)
    print(list)
    for i in list:
        shutil.copy(path+str(i),validation_set_path+"/"+class_i+"/"+copy_path)
def generate():
    dataset = os.listdir(dataset_path)
    for class_i in dataset:
        type = os.listdir(dataset_path+"/"+str(class_i))
        for background in type:
            img_ann = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background))
            imgs = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0]))
            annts = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[1]))
            move_archive(imgs,dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0]),class_i,0)
            move_archive(annts,dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[1]),class_i,1)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",type=int,default=80)
    args = parser.parse_args()
    split_tam = args.split
    generate()
