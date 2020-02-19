import os
dataset_path = "flowchart-3b"
def func(x):
    return x.split(".")[0]

dataset = os.listdir(dataset_path)
for class_i in dataset:
    if(True):
        type = os.listdir(dataset_path+"/"+str(class_i))
        print(class_i)
        for background in type:
            img_ann = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background))
            img_ann = sorted(img_ann)
            imgs = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[1]))
            annts = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[0]))
            a = set(map(func,imgs))
            b = set(map(func,annts))
            print(background)
            print(a-b)
