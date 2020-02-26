import os
import cv2
import xml.etree.ElementTree as ET
dataset_path = "flowchart-3b"

dataset = os.listdir(dataset_path)
for class_i in dataset:
    if(class_i == "arrow_rectangle_down"):
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
                imgi = cv2.imread(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[1])+"/"+imgs[i])
                #cv2.imshow(str(imgs[i]),cv2.resize(imgi, (0,0), fx=0.1, fy=0.1))
                #cv2.waitKey(0)
                for node in parsedXML.getroot().iter('object'):
                    shape = node.find('name').text
                    xmin = int(node.find('bndbox/xmin').text)
                    xmax = int(node.find('bndbox/xmax').text)
                    ymin = int(node.find('bndbox/ymin').text)
                    ymax = int(node.find('bndbox/ymax').text)
                    print(xmin,xmax,ymin,ymax)
                    cv2.imshow(str(imgs[i])+" - "+str(annts[i]),cv2.resize(imgi[ymin:ymax , xmin:xmax],(0,0),fx=0.2, fy=0.2))
                    #cv2.imshow(str(imgs[i])+" - "+str(annts[i]),cv2.resize(imgi[xmax:xmax+xmin , ymax:ymax+ymin],(0,0),fx=0.2, fy=0.2))
                    cv2.waitKey(0)
cv2.waitKey(1)
cv2.destroyAllWindows()
                #df.append(row)
