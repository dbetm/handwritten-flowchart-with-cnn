from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import cv2
"""
Notes:
For now the class have a instance of tkinter, but in the future will be the Handle instance
"""
class Preproccesor(object):
    def __init__(self):
        image = None
        image_path = None
    def open_image(self):
        root = Tk()
        root.config(width=0, height=0)
        root.filename = filedialog.askopenfilename(title = "Select file",filetypes = (("all files","*.*"),("jpeg files","*.jpg"),("png files","*.png")))
        self.image_path = str(root.filename)
        self.image = cv2.imread(self.image_path)
        root.destroy()
    def show_image(self):
        cv2.imshow("image",self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
pre = Preproccesor()
pre.open_image()
pre.show_image()
