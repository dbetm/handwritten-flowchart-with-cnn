import os
import shutil
import xml.etree.ElementTree as ET
dataset_path = "flowchart-3b"


dataset = os.listdir(dataset_path)
for class_i in dataset:
    type = os.listdir(dataset_path+"/"+str(class_i))
    for background in type:
        img_ann = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background))
        img_ann = sorted(img_ann)
        annts = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0]))
        for ann in annts:
            name = ann.split(".")[0]
            tree = ET.parse(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0])+"/"+name+'.xml')
            root = tree.getroot()
            root[2].text = "../images/"+name+".jpg"
            tree.write(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0])+"/"+name+'.xml')
