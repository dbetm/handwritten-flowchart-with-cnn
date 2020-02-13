import matplotlib.pyplot as plt
import cv2
class Segementer(object):
    def __init__(self,image_path,trained_model_path):
        self.image = cv2.imread(image_path)
        self.trained_model_path = trained_model_path
    def segment(self):
        boxes = [[0,400,0,400],[600,800,0,400]]
        images = []
        for box in boxes:
            y1,y2,x1,x2 = box
            crop_img = self.image[y1:y2, x1:x2]
            images.append(crop_img)
        print(images)
seg = Segementer("Images/image.jpeg","path")
seg.segment()
