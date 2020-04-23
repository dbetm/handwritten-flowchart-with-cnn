# -*- coding: utf-8 -*-
import os

import tkinter as tk
from tkinter import ttk
from tkinter import Checkbutton, IntVar
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image

class HandlerGUI(object):

    def __init__(self, master):
        self.master = master
        ##Init of the master view
        self.master.title("Handwritten flowchart with cnn")
        self.master.configure(background="gray99")
        self.master.resizable(False,False)
        self.master.geometry("600x600")
        self.master.config(bg="#857074")
        #Header
        self.header = tk.Frame(self.master)
        self.header.config(width="1000",height="100",bg="#943340")
        self.header.pack(fill="y")
        self.header.pack_propagate(False)
        self.title = tk.Label(self.header,text="3b-flowchart",font=("Arial",50),bg="#943340")
        self.title.pack(pady = 5)
        #Buttons
        btn1 = tk.Button(self.master,height = 4,font = ("Arial",15), width = 25,text = "Train model",command = self.train_window)
        btn1.pack(pady = 80)

        btn2 = tk.Button(self.master,height = 4,font = ("Arial",15), width = 25,text = "Recognize flowchart",command = self.recognize_flowchart_window)
        btn2.pack()
        self.master.mainloop()

    def start_train_action(self,args):
        print(args)
        if(self.__validate(args)):
            print("Train")

    def train_window(self):
        """ Train model window.
        """

        window = tk.Toplevel(self.master)
        window.pack_propagate(False)
        window.title("Train model")
        window.config(width="700", height="600",bg="#943340")
        title = tk.Label(window,font = ("Arial",50),text="Train model",bg="#943340")
        title.pack()
        large_font = ('Arial',15)
        text_font = ('Arial',12)
        mini_text_font = ('Arial',7)
        inputs = tk.Frame(window)
        inputs.config(bg="#943340")
        inputs.pack(side = tk.LEFT)

        # Select folder path of dataset
        text = tk.StringVar()
        dataset_path_text = tk.Label(inputs,height=3, width=50,bg="#943340",font=mini_text_font, textvariable=text)
        dataset_path_text.grid(row=0, column=1)
        dataset_path_button = tk.Button(inputs,text="* Select dataset",font=("Arial",9),width=10,command=lambda : self.__select_dataset_path(text)).grid(row=0,column=0)

        # Pre-trained-model-path
        text_2 = tk.StringVar()
        pretrained_model_path_text = tk.Label(inputs,height=3, width=50,bg="#943340",font=mini_text_font, textvariable=text_2)
        pretrained_model_path_text.grid(row=1, column=1)
        pretrained_model_path_button = tk.Button(inputs,text="Select trained model",font=("Arial",9),width=14,command=lambda : self.__select_pretrained_model_path(text_2)).grid(row=1,column=0)

        # Number of Regions of Interest (RoIs)
        num_rois_text = tk.Label(inputs,text="# RoIs",height=3, width=15,bg="#943340",font=text_font).grid(row=2)
        num_rois_input = tk.Entry(inputs,font=large_font)
        num_rois_input.grid(row=2, column=1)

        # Number of epochs
        num_epochs_text = tk.Label(inputs,text="* Epochs",height=3, width=15,bg="#943340",font=text_font).grid(row=3)
        num_epochs_input = tk.Entry(inputs,font=large_font)
        num_epochs_input.grid(row=3, column=1)

        # Learning rate
        learning_rate_text = tk.Label(inputs,text="* learning rate",height=3, width=15,bg="#943340",font=text_font).grid(row=4)
        learning_rate_input = tk.Entry(inputs,font=large_font)
        learning_rate_input.grid(row=4, column=1)

        # Check - use_gpu
        use_gpu_text = tk.Label(inputs,text="Use GPU",height=3, width=15,bg="#943340",font=text_font).grid(row=5)
        use_gpu_val = IntVar()
        use_gpu_check = Checkbutton(inputs, variable=use_gpu_val)
        use_gpu_check.grid(row=5, column=1)

        #start button
        start_button = tk.Button(
            inputs,
            text="Start",
            font=("Arial",15),
            width=10,
            command=lambda :
                self.start_train_action(
                    [dataset_path_text.cget("text"),num_rois_input.get(), pretrained_model_path_text.cget("text"), num_epochs_input.get(), learning_rate_input.get(), use_gpu_val.get()]
                )
        )
        start_button.grid(row=6,column=1)

        # treeview = ttk.Treeview(window)
        # treeview.pack(side = tk.LEFT,padx = 50)

    def __select_dataset_path(self, label):
        aux = filedialog.askdirectory()
        label.set(aux)

    def __select_pretrained_model_path(self, label):
        aux = filedialog.askopenfilename(
            title = "Select file",
            filetypes = (("hdf5 files","*.hdf5"), ("h5 files","*.h5"))
        )
        label.set(aux)

    def __validate(self, args):
        error_msg = ""

        dataset_path = args[0]
        num_rois = args[1]
        pretrained_model_path = args[2]
        num_epochs = args[3]
        learning_rate = args[4]
        vali = 5 * [True]

        if not(os.path.isdir(dataset_path)):
            vali[0] = False
            error_msg += "Dataset path not valid"
        if(num_rois != ""):
            if(self.__represents_type(num_rois, "int")):
                if(int(num_rois) <= 3):
                    vali[1] = False
                    error_msg += "\nNum rois not valid"
            else:
                vali[1] = False
                error_msg += "\nNum rois must be a integer"
        if(pretrained_model_path != ""):
            if not(os.path.isfile(pretrained_model_path)):
                vali[2] = False
                error_msg += "\nPre-trained model path not valid"
        if(num_epochs != ""):
            if(self.__represents_type(num_epochs, "int")):
                if(int(num_epochs) < 1):
                    vali[3] = False
                    error_msg += "\nNum epochs not valid"
            else:
                vali[3] = False
                error_msg += "\nNum epochs must be a integer"
        if(learning_rate != ""):
            if(self.__represents_type(learning_rate, "float")):
                if(float(learning_rate) <= 0.0):
                    vali[4] = False
                    error_msg += "\nLearning rate not valid"
            else:
                vali[4] = False
                error_msg += "\nLearning rate must be a real number"

        ans = vali[0] and vali[1] and vali[2] and vali[3] and vali[4]
        # Display error message box
        if not(ans):
            messagebox.showerror("Error(s)", error_msg)

        return ans

    def __represents_type(self, var, type):
        if(type == "int"):
            try:
                int(var)
                return True
            except ValueError:
                return False
        elif(type == "float"):
            try:
                float(var)
                return True
            except ValueError:
                return False
        elif(type == "str"):
            try:
                str(var)
                return True
            except ValueError:
                return False
        else:
            return False

    def recognize_flowchart_window(self):
        models_path = "Images"
        """
        Recognize flowchart window
        """
        window = tk.Toplevel(self.master)
        window.pack_propagate(False)
        window.title("Recognize flowchart")
        window.config(width="400", height="350",bg="#943340")
        title_text = tk.Label(window,text="Recognize flowchart",height=3, width=20,bg="#943340",font=("Arial",25))
        title_text.pack()
        #Diferent models to select
        model_list = os.listdir(models_path)
        combobox_model = ttk.Combobox(window,values = model_list,width = 22,font=("Arial",15))
        combobox_model.pack(pady = 10)
        combobox_model.current(0)
        #boton
        button_image = tk.Button(window,text = "select image",width = 20,height=2,font=("Arial",15))
        button_image.pack(pady = 10)
        #button to start to predict
        button_predict = tk.Button(window,text = "predict",width = 20,height=2,font=("Arial",15),command = self.show_results)
        button_predict.pack(pady = 10)

    def show_results(self):
        window = tk.Toplevel(self.master)
        window.pack_propagate(False)
        window.config(width="800", height="620",bg="#943340")
        window.title("Results")
        #title
        title_text = tk.Label(window,text = "Results",height = 3, width = 20, bg = "#943340",font=("Arial",25))
        title_text.pack()
        #code visualtiation
        code_panel = tk.Text(window,width=30,height=21,font=("Arial",15))
        code_panel.pack(side = tk.LEFT,padx = 30)
        code_text = open("results/result1.c",'r')
        count = 0
        while True:
            count += 1
            line = code_text.readline()
            if not line:
                break
            code_panel.insert(tk.INSERT,line)
        #image
        img = Image.open("results/result1.jpg")
        img.thumbnail((500,500), Image.ANTIALIAS)
        imgL = ImageTk.PhotoImage(img)
        panel = tk.Label(window,image = imgL)
        panel.image = imgL
        panel.pack(side = tk.LEFT)

root = tk.Tk()
my_gui = HandlerGUI(root)
