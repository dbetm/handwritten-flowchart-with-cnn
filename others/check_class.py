import os
import xml.etree.ElementTree as ET
dataset_path = "flowchart-3b"
dataset = os.listdir(dataset_path)
for class_i in dataset:
    if(True):
        type = os.listdir(dataset_path+"/"+str(class_i))
        for background in type:
            img_ann = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background))
            img_ann = sorted(img_ann)
            imgs = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[1]))
            annts = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0]))
            imgs = sorted(imgs)
            annts = sorted(annts)
            for i in range(len(annts)):
                parsedXML = ET.parse(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0])+"/"+annts[i])
                #cv2.imshow(str(imgs[i]),cv2.resize(imgi, (0,0), fx=0.1, fy=0.1))
                #cv2.waitKey(0)
                for node in parsedXML.getroot().iter('object'):
                    shape = node.find('name').text
                    if(shape != class_i):
                        print("ERROR",class_i,shape)
                        
