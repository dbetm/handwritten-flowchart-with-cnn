import tkinter as tk
from tkinter import ttk
import os
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
        ans = (sum(int(x.isdigit()) for x in args) == len(args))
        if(ans):
            print("Entrenar")
    def train_window(self):
        """
        Train model window
        """
        window = tk.Toplevel(self.master)
        window.pack_propagate(False)
        window.title("Train model")
        window.config(width="700", height="600",bg="#943340")
        title = tk.Label(window,font = ("Arial",50),text="Train model",bg="#943340")
        title.pack()
        large_font = ('Arial',15)
        text_font = ('Arial',12)
        inputs = tk.Frame(window)
        inputs.config(bg="#943340")
        inputs.pack(side = tk.LEFT)
        #bath size
        bath_size_text = tk.Label(inputs,text="Batch size",height=3, width=15,bg="#943340",font=text_font).grid(row = 0)
        bath_size_input = tk.Entry(inputs,font=large_font)
        bath_size_input.grid(row=0, column=1)
        #learning rate
        learning_rate_text = tk.Label(inputs,text="learning rate",height=3, width=15,bg="#943340",font=text_font).grid(row = 1)
        learning_rate_input = tk.Entry(inputs,font=large_font)
        learning_rate_input.grid(row=1, column=1)
        #img row
        img_row_text = tk.Label(inputs,text="img_row",height=3, width=15,bg="#943340",font=text_font).grid(row = 2)
        img_row_input = tk.Entry(inputs,font=large_font)
        img_row_input.grid(row=2, column=1)
        #img cols
        img_cols_text = tk.Label(inputs,text="img_cols",height=3, width=15,bg="#943340",font=text_font).grid(row = 3)
        img_cols_input = tk.Entry(inputs,font=large_font)
        img_cols_input.grid(row=3, column=1)
        #num_train
        num_train_text = tk.Label(inputs,text="num_train",height=3, width=15,bg="#943340",font=text_font).grid(row = 4)
        num_train_input = tk.Entry(inputs,font=large_font)
        num_train_input.grid(row=4, column=1)
        #num validation
        num_validation_text = tk.Label(inputs,text="num_validation",height=3, width=15,bg="#943340",font=text_font).grid(row = 5)
        num_validation_input = tk.Entry(inputs,font=large_font)
        num_validation_input.grid(row=5, column=1)
        #start button
        start_button = tk.Button(inputs,text="Start",font=("Arial",15),width=10,command=lambda : self.start_train_action([bath_size_input.get(),learning_rate_input.get(),img_row_input.get(),img_cols_input.get(),num_train_input.get(),num_validation_input.get()])).grid(row=6,column=1)

        treeview = ttk.Treeview(window)
        treeview.pack(side = tk.LEFT,padx = 50)

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
