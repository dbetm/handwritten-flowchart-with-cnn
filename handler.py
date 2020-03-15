import tkinter as tk
class HandlerGUI(object):
    def __init__(self, master):
        self.master = master
        ##Init of the master view
        self.master.title("Handwritten flowchart with cnn")
        self.master.configure(background="gray99")
        self.master.resizable(False,False)
        self.master.geometry("1000x600")
        self.master.config(bg="#857074")
        #Header
        self.header = tk.Frame(self.master)
        self.header.config(width="1000",height="100",bg="#943340")
        self.header.pack(fill="y")
        self.header.pack_propagate(False)
        self.title = tk.Label(self.header,text="3b-flowchart",font=("Arial",50),bg="#943340")
        self.title.pack(pady = 5)
        #Buttons
        btn1 = tk.Button(self.master,height=4,font=("Arial",15), width=25,text="Train model")
        btn1.pack(pady = 80)

        btn2 = tk.Button(self.master,height=4,font=("Arial",15), width=25,text="Recognize flowchart")
        btn2.pack()
        self.master.mainloop()
root = tk.Tk()
my_gui = HandlerGUI(root)
