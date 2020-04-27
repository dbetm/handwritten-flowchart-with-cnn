import os
import keras
from dataGenerator import DataGenerator
from keras.models import Sequential
#Dense Means fully connected layers , Dropout is a technique to improve convergence ,
# Flatten is to reshape the matrix for giving to different layers
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Input, Activation, ZeroPadding2D
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm
import cv2
from data_preparation import Preproc
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
tf.config.experimental.list_physical_devices(device_type=None)
class Model:
    def __init__(self):
        self.num_classes = 62+23
        self.input_shape = (28,28,1)
        self.batch_size = 128
        self.epochs = 100
        self.model = None
        self.output_path = "output"
        self.target_path = self.output_path + "/checkpoint_weights.hdf5"
        f = open("data/mapping.txt", "r")
        lines = []
        left,rigth = [],[]
        for i in f:
            lines.append(i[0:len(i)-1])
        for i in lines:
            res = i.split(" ")
            left.append(int(res[0]))
            rigth.append(list(map(chr,map(int,res[1].split(",")))))
        self.map = dict(zip(left,rigth))
        print(self.map)
    def get_model_1(self):
        model = Sequential()
        #convolutional layer with rectified linear unit activation
        model.add(Conv2D(64, kernel_size=(3,3),activation='relu',input_shape=self.input_shape))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        #choose the best features via pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #randomly turn neurons on and off to improve convergence
        model.add(Dropout(0.25))
        #flatten since too many dimensions, we only want a classification output
        model.add(Flatten())
        #fully connected to get all relevant data
        model.add(Dense(128, activation='relu'))
        #one more dropout for convergence' sake :)
        model.add(Dropout(0.5))
        #output a softmax to squash the matrix into output probabilities
        model.add(Dense(self.num_classes, activation='softmax'))
        return model
    def get_model_2(self):
        X_input = Input(self.input_shape)
        # zero padding probably not required since the main digit is in the centre only
        # X = zeroPadding2D((1,1))(X_input)

        X = Conv2D(32,(3,3),strides = (1,1), name = 'conv0')(X_input)
        X = BatchNormalization(axis=3,name='bn0')(X)
        X = Activation('relu')(X)
        X = Conv2D(32,(3,3),strides = (1,1), name = 'conv1')(X)
        X = BatchNormalization(axis=3,name='bn1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2,2),strides = (2,2),name = 'MP1')(X)

        X = Conv2D(64,(3,3),strides = (1,1), name = 'conv2')(X)
        X = BatchNormalization(axis=3,name='bn2')(X)
        X = Activation('relu')(X)
        X = Conv2D(64,(3,3),strides = (1,1), name = 'conv3')(X)
        X = BatchNormalization(axis=3,name='bn3')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2,2),strides = (2,2),name = 'MP2')(X)

        X = Dropout(0.2)(X)
        X = Flatten()(X)
        X = Dense(256,activation = 'relu',name= 'fc1')(X)
        X = Dropout(0.4)(X)
        X = Dense(self.num_classes,activation = 'softmax',name = 'fco')(X)

        model = Model(inputs = X_input,outputs = X, name = 'MNIST_Model')
    def get_model_3(self):
        model = Sequential()

        model.add(Conv2D(32,(3,3),strides = (1,1),input_shape=self.input_shape))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(Conv2D(32,(3,3),strides = (1,1)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),strides = (2,2)))

        model.add(Conv2D(64,(3,3),strides = (1,1)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(Conv2D(64,(3,3),strides = (1,1)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),strides = (2,2)))

        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(256,activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes,activation = 'softmax'))

        return model

    def separete_characteres(self,image):
        cv2.imshow("Imagen",image)
        cv2.waitKey(0)
        h,w = image.shape
        print("Img tam",w,h)
        val = None
        TF_list = []
        for x in range(w):
            for y in range(h):
                if(image[y,x] == 255):
                    val = True
                    break
                else:
                    val = False
            TF_list.append(val)
        flag = False
        pos = 0
        pp = Preproc()
        res = ""
        for i in range(1,len(TF_list)):
            if(TF_list[i] == True and TF_list[i-1] == False):
                pos = i
            if(TF_list[i] == False and TF_list[i-1] == True):
                img = pp.resize(image[0:h,pos:i])
                #cv2.imshow("imagen",img)
                #cv2.waitKey(0)
                res += self.model_predict(img)
        print("Resultado",res)

    def model_predict(self,image):
        self.model = self.get_model_3()
        #Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
        #categorical ce since we have multiple classes (10)
        #train that
        # get default callbacks and load checkpoint weights file (HDF5) if exists
        model_path = "output/models/5/checkpoint_weights.hdf5"
        if os.path.isfile(model_path):
            self.model.load_weights(model_path)
        image = image.reshape(1,28,28,1)
        print("Shape = ",image.shape)
        predicts = self.model.predict_classes(x=image)
        res = ""
        for i in self.map[predicts[0]]:
            res += str(i)
        print("(res)",res)
        return res
        #print(keras.utils.to_categorical(predicts,self.num_classes))
    def train(self):
        self.model = self.get_model_3()
        self.model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

        #------------------------( Get the data )------------------------------------
        dataGenerator = DataGenerator("data/mnist/gzip")
        x_train,y_train,x_test, y_test = dataGenerator.get_data()
        print("-----------------------------------------------")
        print("shape",x_train.shape)
        print("-----------------------------------------------")
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        """,keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=1, mode='auto', baseline=None, restore_best_weights=True)"""
        print(self.model.summary())
        self.model.fit(x_train, y_train,batch_size=self.batch_size,epochs=self.epochs,verbose=1,validation_data=(x_test, y_test),callbacks = [keras.callbacks.callbacks.ModelCheckpoint(self.target_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)])

        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
model = Model()
model.train()
#model.model_predict(cv2.imread("pruebas/1.jpg",0))
"""img = cv2.imread("pruebas/hola.jpg",0)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img = (255 - img)
model.separete_characteres(img)"""
