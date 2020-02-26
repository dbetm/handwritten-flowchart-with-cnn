import os
import shutil
import argparse
from random import randint

dataset_path = "/home/david/datasets/flowchart-3b"
training_set_path = "/home/david/datasets/flowchart-3b(splitter)/train"
validation_set_path = "/home/david/datasets/flowchart-3b(splitter)/validation"
dataset_name = "/home/david/datasets/flowchart-3b(splitter)"
split_tam = 80
count = 0

def create_dataset_splitter():
    os.mkdir(dataset_name)
    os.mkdir(dataset_name+"/train")
    os.mkdir(dataset_name+"/validation")
    os.mkdir(dataset_name+"/train"+"/start_end");os.mkdir(dataset_name+"/train"+"/start_end/images");os.mkdir(dataset_name+"/train"+"/start_end/annots")
    os.mkdir(dataset_name+"/train"+"/scan");os.mkdir(dataset_name+"/train"+"/scan/images");os.mkdir(dataset_name+"/train"+"/scan/annots")
    os.mkdir(dataset_name+"/train"+"/process");os.mkdir(dataset_name+"/train"+"/process/images");os.mkdir(dataset_name+"/train"+"/process/annots")
    os.mkdir(dataset_name+"/train"+"/decision");os.mkdir(dataset_name+"/train"+"/decision/images");os.mkdir(dataset_name+"/train"+"/decision/annots")
    os.mkdir(dataset_name+"/train"+"/print");os.mkdir(dataset_name+"/train"+"/print/images");os.mkdir(dataset_name+"/train"+"/print/annots")
    os.mkdir(dataset_name+"/train"+"/arrow_line_up");os.mkdir(dataset_name+"/train"+"/arrow_line_up/images");os.mkdir(dataset_name+"/train"+"/arrow_line_up/annots")
    os.mkdir(dataset_name+"/train"+"/arrow_line_right");os.mkdir(dataset_name+"/train"+"/arrow_line_right/images");os.mkdir(dataset_name+"/train"+"/arrow_line_right/annots")
    os.mkdir(dataset_name+"/train"+"/arrow_line_down");os.mkdir(dataset_name+"/train"+"/arrow_line_down/images");os.mkdir(dataset_name+"/train"+"/arrow_line_down/annots")
    os.mkdir(dataset_name+"/train"+"/arrow_line_left");os.mkdir(dataset_name+"/train"+"/arrow_line_left/images");os.mkdir(dataset_name+"/train"+"/arrow_line_left/annots")
    os.mkdir(dataset_name+"/train"+"/arrow_rectangle_up");os.mkdir(dataset_name+"/train"+"/arrow_rectangle_up/images");os.mkdir(dataset_name+"/train"+"/arrow_rectangle_up/annots")
    os.mkdir(dataset_name+"/train"+"/arrow_rectangle_right");os.mkdir(dataset_name+"/train"+"/arrow_rectangle_right/images");os.mkdir(dataset_name+"/train"+"/arrow_rectangle_right/annots")
    os.mkdir(dataset_name+"/train"+"/arrow_rectangle_down");os.mkdir(dataset_name+"/train"+"/arrow_rectangle_down/images");os.mkdir(dataset_name+"/train"+"/arrow_rectangle_down/annots")
    os.mkdir(dataset_name+"/train"+"/arrow_rectangle_left");os.mkdir(dataset_name+"/train"+"/arrow_rectangle_left/images");os.mkdir(dataset_name+"/train"+"/arrow_rectangle_left/annots")

    os.mkdir(dataset_name+"/validation"+"/start_end");os.mkdir(dataset_name+"/validation"+"/start_end/images");os.mkdir(dataset_name+"/validation"+"/start_end/annots")
    os.mkdir(dataset_name+"/validation"+"/scan");os.mkdir(dataset_name+"/validation"+"/scan/images");os.mkdir(dataset_name+"/validation"+"/scan/annots")
    os.mkdir(dataset_name+"/validation"+"/process");os.mkdir(dataset_name+"/validation"+"/process/images");os.mkdir(dataset_name+"/validation"+"/process/annots")
    os.mkdir(dataset_name+"/validation"+"/decision");os.mkdir(dataset_name+"/validation"+"/decision/images");os.mkdir(dataset_name+"/validation"+"/decision/annots")
    os.mkdir(dataset_name+"/validation"+"/print");os.mkdir(dataset_name+"/validation"+"/print/images");os.mkdir(dataset_name+"/validation"+"/print/annots")
    os.mkdir(dataset_name+"/validation"+"/arrow_line_up");os.mkdir(dataset_name+"/validation"+"/arrow_line_up/images");os.mkdir(dataset_name+"/validation"+"/arrow_line_up/annots")
    os.mkdir(dataset_name+"/validation"+"/arrow_line_right");os.mkdir(dataset_name+"/validation"+"/arrow_line_right/images");os.mkdir(dataset_name+"/validation"+"/arrow_line_right/annots")
    os.mkdir(dataset_name+"/validation"+"/arrow_line_down");os.mkdir(dataset_name+"/validation"+"/arrow_line_down/images");os.mkdir(dataset_name+"/validation"+"/arrow_line_down/annots")
    os.mkdir(dataset_name+"/validation"+"/arrow_line_left");os.mkdir(dataset_name+"/validation"+"/arrow_line_left/images");os.mkdir(dataset_name+"/validation"+"/arrow_line_left/annots")
    os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_up");os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_up/images");os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_up/annots")
    os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_right");os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_right/images");os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_right/annots")
    os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_down");os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_down/images");os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_down/annots")
    os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_left");os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_left/images");os.mkdir(dataset_name+"/validation"+"/arrow_rectangle_left/annots")
def move_archive(images,annots,path,class_i):
    print("[OK]",class_i)
    global count
    images = sorted(images)
    annots = sorted(annots)
    global split_tam
    num_files = len(images)
    #Calculate the number of files that should be moved to the training set
    num_split = int(num_files * split_tam / 100)
    print()
    for i in range(num_split):
        pos = randint(0,len(images)-1)
        shutil.copy(path+"/images/"+str(images[pos]),training_set_path+"/"+class_i+"/images/"+str(count)+".jpg")
        shutil.copy(path+"/annots/"+str(annots[pos]),training_set_path+"/"+class_i+"/annots/"+str(count)+".xml")
        count = count + 1
        images.pop(pos)
        annots.pop(pos)
    count_aux = count
    for i in images:
        shutil.copy(path+"/images/"+str(i),validation_set_path+"/"+class_i+"/images/"+str(count_aux)+".jpg")
        count_aux = count_aux + 1
    count_aux = count
    for i in annots:
        shutil.copy(path+"/annots/"+str(i),validation_set_path+"/"+class_i+"/annots/"+str(count_aux)+".xml")
        count_aux = count_aux + 1
    count = count_aux
def generate():
    dataset = os.listdir(dataset_path)
    for class_i in dataset:
        type = os.listdir(dataset_path+"/"+str(class_i))
        for background in type:
            img_ann = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background))
            img_ann = sorted(img_ann)
            imgs = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[1]))
            annts = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0]))
            move_archive(imgs,annts,dataset_path+"/"+str(class_i)+"/"+str(background)+"/",class_i)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",type=int,default=80)
    args = parser.parse_args()
    split_tam = args.split
    create_dataset_splitter()
    generate()
    print("Archives copy :",count*2)
