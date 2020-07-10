# -*- coding: utf-8 -*-
import os
import json
import sys

import tkinter as tk
from tkinter import ttk
from tkinter import Checkbutton, IntVar
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Style
from PIL import ImageTk, Image
import cv2

from node import Node
from graph import Graph
from codeGenerator import CodeGenerator
from text_model.data.reader import Dataset
from text_model.text_classifier import TextClassifier
from model.shape_classifier import ShapeClassifier
from flowchart_generator.flowchart_generator import FlowchartGenerator
new = "new_data"
new_source_path = os.path.join("text_model","data_model",f"{new}.hdf5")
class VerticalScrolledFrame(tk.Frame):
    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)
        vscrollbar = tk.Scrollbar(self, orient=tk.VERTICAL,width=20)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.TRUE)
        canvas = tk.Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set,height = 500)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        vscrollbar.config(command=canvas.yview)
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)
        self.interior = interior = tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=tk.NW)
        def _configure_interior(event):
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)
        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)
        interior.bind_all('<MouseWheel>', lambda event:     vscrollbar.yview("scroll",event.delta,"units"))
class HandlerGUI(object):

    def __init__(self, master, env_name):
        self.ds = Dataset(source="", name="iam")
        self.RESULTS_PATH = "results/"
        self.master = master
        ##Init of the master view
        self.master.title("Handwritten flowchart with CNNs")
        self.master.configure(background="gray99")
        self.master.resizable(False,False)
        self.master.geometry("600x500")
        self.master.config(bg="#857074")
        # Predict
        self.selected_image = ""
        self.models_path = "model/training_results/"
        #training
        self.env_name = env_name
        #Header
        self.header = tk.Frame(self.master)
        self.header.config(width="1000",height="100",bg="#943340")
        self.header.pack(fill="y")
        self.header.pack_propagate(False)
        self.title = tk.Label(self.header,text="3b-flowchart",font=("Arial",50),bg="#943340")
        self.title.pack(pady = 20)
        style = Style()
        style.map('TButton', foreground = [('active', 'green')],background = [('active', 'black')])
        #Buttons
        btn1 = tk.Button(self.master,height = 4,font = ("Arial",15), width = 25,text = "Train shape model",command = self.train_window)
        btn1.pack(pady = 10)

        btn2 = tk.Button(self.master,height = 4,font = ("Arial",15), width = 25,text = "Recognize flowchart",command = self.recognize_flowchart_window)
        btn2.pack(pady = 10)

        btn3 = tk.Button(self.master,height = 4,font = ("Arial",15), width = 25,text = "Train text model",command = self.__train_text_model)
        btn3.pack(pady = 10)
        self.master.mainloop()

    def start_train_action(self,args):
        if(self.__validate_train_inputs(args)):
            dataset_path = args[0]
            rois = None if args[1] == '' else int(args[1])
            weights_input = None if args[2] == '' else args[2]
            epochs = None if args[3] == '' else int(args[3])
            lr = None if args[4] == '' else float(args[4])
            use_gpu = True if args[5] == 1 else False

            cmd = 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate ' + self.env_name
            cmd += ' && cd model/ && python3 shape_model.py; exec bash'

            args = {
                "dataset": dataset_path,
                "rois": rois,
                "input_weight_path": weights_input,
                "epochs": epochs,
                "learning_rate": lr,
                "gpu": use_gpu
            }
            with open('model/args.json', 'w') as json_file:
                json.dump(args, json_file)
                json_file.close()

            cmd_exec = "gnome-terminal -e 'bash -c \"" + cmd + "\"'"
            # cmd_exec = "gnome-terminal -e 'bash -c \"conda init bash && python3 test.py; exec bash\"'"
            print(cmd_exec)
            os.system(cmd_exec)


    def train_window(self):
        """ Train model window.
        """

        window = tk.Toplevel(self.master)
        window.pack_propagate(False)
        window.title("Train shape model")
        window.config(width="700", height="600",bg="#943340")
        title = tk.Label(window,font = ("Arial",50),text="Train shape model",bg="#943340")
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

    def __select_image(self):
        self.selected_image = filedialog.askopenfilename(
            title="Select image",
            filetypes=(
                ("all files","*.*"),
                ("jpeg files",("*.jpg, *.jpeg")),
                ("png files","*.png")
            )
        )

    def __validate_train_inputs(self, args):
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

    def __validate_predict_inputs(self, args):
        error_msg = ""
        folder_training_results = args[0]
        image_path = args[1]
        # args[2] => use_gpu, domain limited, not necessary validation
        num_rois = args[3]
        if(num_rois == "Type number of RoIs"):
            num_rois = "32"
        vali = 4 * [True]

        model_path = self.models_path + folder_training_results
        # Validation
        # Path of model
        if not(os.path.isdir(model_path)):
            vali[0] = False
            error_msg += "Training results folder not valid"
        else:
            file = self.__search_model(model_path)
            if(file == "-1"):
                vali[0] = False
                error_msg += "Training results folder not contains any model"
        # Image path
        if(os.path.isfile(image_path)):
            if('.' in image_path):
                format = image_path.split(".")
                format = format[len(format)-1]
                if(format != 'png' and format != 'jpg' and format != 'jpeg'):
                    vali[1] = False
                    error_msg += "\nFormat image not valid"
            else:
                vali[1] = False
                error_msg += "\nFormat image not valid"
        else:
            vali[1] = False
            error_msg += "\nImage not found"
        # Number of Regions of Interest (RoIs)
        if(num_rois != ""):
            if(self.__represents_type(num_rois, "int")):
                if(int(num_rois) <= 3):
                    vali[1] = False
                    error_msg += "\nNum rois not valid"
            else:
                vali[1] = False
                error_msg += "\nNum rois must be a integer"

        ans = vali[0] and vali[1] and vali[3]
        # Display error message box
        if not(ans):
            messagebox.showerror("Error(s)", error_msg)

        return ans

    def __search_model(self, model_path):
        files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
        validated_file = ""
        for file in files:
            if('.' in file):
                format = file.split(".")
                format = format[len(format)-1]
                if(format == 'hdf5' or format == 'h5'):
                    validated_file = file
                    break
        if(validated_file == ""):
            return "-1"
        else:
            # Return the first model founded
            return validated_file

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

    def __get_results_path(self):
        results_dir = os.listdir(self.RESULTS_PATH)
        n = len(results_dir) + 1

        while(True):
            new_dir = "results_"+str(n) + "/"
            if(os.path.isdir(self.RESULTS_PATH + new_dir)):
                n += 1
            else:
                break

        print('Results will be stored in: ' + new_dir)
        return new_dir

    def recognize_flowchart_window(self):
        """ Recognize flowchart window.
        """

        window = tk.Toplevel(self.master)
        header = tk.Frame(window)
        header.config(width="400",height="50",bg="#857074")
        header.pack(fill="y")
        header.pack_propagate(False)
        title = tk.Label(header,text="Recognize flowchart",font=("Arial",20),bg="#857074")
        title.pack(pady = 5)
        window.pack_propagate(False)
        window.config(width="400", height="470",bg="#943340")

        # Diferent models to select
        model_folder_list = os.listdir(self.models_path)
        model_folder_list.append("Select a folder of training results")
        model_folder_list.reverse()
        combobox_model_folder = ttk.Combobox(window,values = model_folder_list, width =27,font=("Arial",13))
        combobox_model_folder.pack(pady=40)
        combobox_model_folder.current(0)

        # Button for select image
        button_image = tk.Button(window,text="Select image",width=18,height=2,font=("Arial",12), command=self.__select_image)
        button_image.pack(pady=10)

        # Use GPU
        use_gpu_val = IntVar()
        use_gpu_check = Checkbutton(window, text="Use GPU", variable=use_gpu_val,width=20,height=2,background="#943340")
        use_gpu_check.pack(pady=10)

        # Number of RoIs
        num_rois_lbl = tk.Label(window,text="Optional, default: 32",height=2, width=20,bg="#943340",font=("Arial",10))
        num_rois_lbl.pack()
        num_rois_input = tk.Entry(window,font=("Arial",12), width=20)
        num_rois_input.insert(0, 'Type number of RoIs')
        num_rois_input.bind('<FocusIn>', lambda args: num_rois_input.delete('0', 'end'))
        num_rois_input.bind('<FocusOut>', lambda x: num_rois_input.insert('0', 'Type number of RoIs') if not num_rois_input.get() else 0)
        num_rois_input.pack(pady=10)

        # Button for start to predict
        button_predict = tk.Button(
            window,
            text="Predict",
            width=20,
            height=2,
            font=("Arial",15),
            background="green",
            command=lambda :
                self.predict(
                    [
                        combobox_model_folder.get(),
                        self.selected_image,
                        use_gpu_val.get(),
                        num_rois_input.get()
                    ],window
                )
        )
        button_predict.pack(pady=20)
    def __continue_process(self,text_nodes,shape_nodes,image_path,window):
        if window != None:
            window.destroy()
            window.update()
        graph = Graph(text_nodes,shape_nodes)
        flow = graph.generate_graph()
        #call function to traslate to code and flowchart
        results_path = self.__get_results_path()
        os.mkdir(self.RESULTS_PATH+results_path)
        cg = CodeGenerator(graph,results_path)
        cg.generate_code(graph.find_first_state(),-1)
        fg = FlowchartGenerator(graph,flow,results_path)
        fg.generate_flowchart()
        self.show_results(results_path)
    def __train_text_model(self):
        if(os.path.isfile(new_source_path)):
            tc = TextClassifier()
            tc.train_new_data()
            tc = None
        else:
            messagebox.showerror("Error", "there are no data to train model")
    def __train_now(self,images,words,text_nodes,shape_nodes,image_path,window):
        window.destroy()
        window.update()
        self.ds.save_new_data(images,words)
        self.tc.train_new_data()
        os.remove(new_source_path)
        self.__continue_process(text_nodes,shape_nodes,image_path,None)
    def __train_or_show(self,new_texts,text_nodes,shape_nodes,image_path,images,window):
        window.destroy()
        window.update()
        for node,text in zip(text_nodes,new_texts):
            node.set_text(text)
        window = tk.Toplevel(self.master)
        header = tk.Frame(window)
        header.config(width="400",height="50",bg="#857074")
        header.pack(fill="y")
        header.pack_propagate(False)
        title = tk.Label(header,text="",font=("Arial",20),bg="#857074")
        title.pack(pady = 5)
        window.pack_propagate(False)
        window.config(width="400", height="200",bg="#943340")
        train_now = tk.Button(window,text="Train text model now",font=("Arial",15),
        background="green",command = lambda:self.__train_now(images,new_texts,text_nodes,shape_nodes,image_path,window))
        train_now.pack(pady = 10)
        train_after = tk.Button(window,text="Train text model later",font=("Arial",15),
        background="green",command = lambda:self.__continue_process(text_nodes,shape_nodes,image_path,window))
        train_after.pack(pady = 10)
    def edit_text(self,text_nodes,shape_nodes,image_path,window):
        window.destroy()
        window.update()
        text_nodes = list(text_nodes)
        window = tk.Toplevel(self.master)
        header = tk.Frame(window)
        header.config(width="400",height="50",bg="#857074")
        header.pack(fill="y")
        header.pack_propagate(False)
        title = tk.Label(header,text="Edit text",font=("Arial",20),bg="#857074")
        title.pack(pady = 5)
        message = tk.Label(window,text="If text prediction are not aceptable please edit it",font=("Arial",12),bg="#857074")
        message.pack(pady = 5)
        window.pack_propagate(False)
        window.config(width="400", height="600",bg="#943340")
        entrys = []
        aux = [x[0] for x in text_nodes]
        imgs = [x[1] for x in text_nodes]
        continue_btn = tk.Button(window,text = "Continue",background="green",command = lambda: self.__train_or_show([x.get() for x in entrys],aux,shape_nodes,image_path,imgs,window))
        continue_btn.pack()
        display = tk.Frame(window)
        display.pack(fill="x",pady=10,side=tk.LEFT,anchor=tk.N)
        display.config(bg="#454545",width="700",height="400")
        scframe = VerticalScrolledFrame(display)
        scframe.pack()
        for node in text_nodes:
            index = tk.Frame(scframe.interior,relief = tk.RAISED)
            index.pack(pady = 15)
            imgL = ImageTk.PhotoImage(image = Image.fromarray(node[1]))
            img = tk.Label(index,image = imgL)
            img.image = imgL
            img.pack()
            txt = tk.Entry(index, width=100,font="Arial")
            entrys.append(txt)
            txt.insert(0,str(node[0].get_text()))
            txt.pack()
    def predict(self, args,window):
        if(self.__validate_predict_inputs(args)):
            #model = self.__search_model(self.models_path + args[0])
            model = self.models_path + args[0]
            image_path = args[1]
            use_gpu = True if args[2] else False
            num_rois = 32 if args[3] == 'Type number of RoIs' else int(args[3])
            #Get the image
            image = cv2.imread(image_path)
            #Text segmentation(areas)
                #Text predict(text value)
                #[Node..........]
            sc = ShapeClassifier(results_path = model,
            use_gpu=use_gpu,
            num_rois=num_rois,
            bbox_threshold=0.51,
            overlap_thresh_1=0.9,
            overlap_thresh_2=0.2)
            shape_nodes = sc.predict(image,display_image=True)
            #build the graph
            self.tc = TextClassifier()
            text_nodes = self.tc.recognize(image_path)
            window.destroy()
            window.update()
            #collapse text nodes
            self.edit_text(text_nodes,shape_nodes,image_path,window)

    def show_results(self,results_path):
        window = tk.Toplevel(self.master)
        header = tk.Frame(window)
        header.config(width="820",height="80",bg="#857074")
        header.pack(fill="y")
        header.pack_propagate(False)
        title = tk.Label(header,text="Results",font=("Arial",50),bg="#857074")
        title.pack(pady = 5)
        window.pack_propagate(False)
        window.config(width="820", height="660",bg="#943340")
        window.title("Results")

        #code visualtiation
        code_panel = tk.Text(window,width=30,height=21,font=("Arial",15),bg="#ccc5c3")
        code_panel.pack(side = tk.LEFT,padx = 30)
        code_text = open("results/"+results_path+"code.c",'r')
        count = 0
        while True:
            count += 1
            line = code_text.readline()
            if not line:
                break
            code_panel.insert(tk.INSERT,line)
        code_panel.config(state=tk.DISABLED)
        #image
        img = Image.open("results/"+results_path+"flowchart.png")
        img = img.resize((400,500), Image.ANTIALIAS)
        imgL = ImageTk.PhotoImage(img)
        panel = tk.Label(window,image = imgL)
        panel.image = imgL
        panel.pack(side = tk.LEFT)
        # Compile code source
        filepath = 'results/' + results_path + "code.c"
        objectpath = 'results/' + results_path + 'code.o'
        os.system('gcc -Wall ' + filepath + ' -o ' + objectpath)
        os.system('echo "Compilation done!"')


root = tk.Tk()
# hf is the name of the Conda environment
my_gui = HandlerGUI(root, "tt")
