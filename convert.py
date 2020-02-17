import os
dataset_path = "flowchart-3b"
dataset = os.listdir(dataset_path)
for class_i in dataset:
    type = os.listdir(dataset_path+"/"+str(class_i))
    for background in type:
        img_ann = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background))
        imgs = os.listdir(dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[1]))
        path = dataset_path+"/"+str(class_i)+"/"+str(background)+"/"+str(img_ann[1])
        #print(imgs)
        for img in imgs:
            ans = "convert "+path+"/"+str(img)+" "+path+"/"+str(img).split(".")[0]+".jpg"
            #print(ans)
            #os.rename(path+"/"+str(img),path+"/"+str(img).split(".")[0].replace(' ','')+"."+str(img).split(".")[1])
            #os.system(ans)
            for img in imgs:
                if(str(img).split(".")[1] != "jpg"):
                    print("ERROR")
