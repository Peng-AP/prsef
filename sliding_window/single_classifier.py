import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization,Flatten, Dropout
import glob
import pickle
from sklearn.model_selection import train_test_split
from utils import squarify
import math
import keras.datasets


"""Data"""
num_classes = 5
image_dirs = "C:\\Users\\AaronPeng\\Desktop\\PRSEF\\object-dataset\\"
classes = {
    'background' : 0,
    'pedestrian' : 1,
    'automobile' : 2,
    'truck' : 3
}

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

realx = []
realy = []
nt = 0
for i in range(50000):
    if(len(realx) == 1000): break
    if(y_train[i] == 1):
        realx.append(x_train[i])
        realy.append(2)
    elif(y_train[i] == 9):
        nt += 1
        realx.append(x_train[i])
        realy.append(3)
print(nt)
print(".")
peds = []
npeds = []
for i in range(500):
    pimg = cv2.imread(f"C:\\Users\\AaronPeng\\Downloads\\DC-ped-dataset_base.tar\\1\\ped_examples\\img_{'0'*(5-len(str(i)))}{i}.pgm")
    npimg = cv2.imread(f"C:\\Users\\AaronPeng\\Downloads\\DC-ped-dataset_base.tar\\1\\non-ped_examples\\img_{'0'*(5-len(str(i)))}{i}.pgm")
    peds.append(pimg)
    npeds.append(npimg)


output = open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\pedinputs.pkl",'wb')
pickle.dump(peds,output)
output.close() 
output = open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\npedinputs.pkl",'wb')
pickle.dump(npeds,output)
output.close()
output = open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\cifarX.pkl",'wb')
pickle.dump(realx,output)
output.close()
output = open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\cifarY.pkl",'wb')
pickle.dump(realy,output)
output.close()


"""anns = pd.read_csv("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\labels.csv").to_numpy()
d = {}
annotations = []
inputs = []
labels = []
coordinateLabels = []
for i in range(len(anns)):
    box = anns[i][0].split(" ") 
    annotations.append([box[0],int(box[1]),int(box[2]), int(box[3]), int(box[4]), box[5], box[6], int(box[3])-int(box[1]),int(box[4])-int(box[2])]) #add width + height
    if(annotations[i][0] in d):
        d[annotations[i][0]].append(annotations[i])
    else: d[annotations[i][0]] = [annotations[i]]

counter = 0
for key in d.keys():    
    img = cv2.resize(cv2.imread(f"{image_dirs}{key}",0),(480,300))
    for ann in d[key]:
        file,left,top,right,bottom,s,label,width,height = ann
        pad = abs((width-height)//2)
        if(width>height):
            final_top = max((top - pad)//4,0)
            final_bottom = min((bottom + pad)//4, 300)
            final_left, final_right = left//4,right//4
        elif(height>width):
            final_left = max((left - pad)//4,0)
            final_right = min((right + pad)//4, 480)
            final_top, final_bottom = top//4,bottom//4
        sec = np.expand_dims(img[final_top:final_bottom,final_left:final_right], -1)
        chunk = squarify(sec,(50,50))
        inputs.append(np.expand_dims(chunk,-1))
        coordinateLabels.append([left//4,top//4,width//4,height//4])
    counter+=1
    if(counter%100 == 0): print(counter)
output = open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\realimageinputs.pkl",'wb')
pickle.dump(inputs,output)
output.close() 
output = open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\coordlabels.pkl",'wb')
pickle.dump(coordinateLabels,output)
output.close()""" 



#for dir in glob.glob(image_dirs):
    #img = cv2.resize(cv2.imread(dir),(192,120))
    #print(img.shape)
    #cv2.imshow("img",img)
    #cv2.waitKey()

""" with open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\grey_4inputs.pkl",'rb') as pkl: inputs = pickle.load(pkl)
for i in range(len(inputs)):
    h,w,c = inputs[i].shape
    if(h > w): inputs[i] = cv2.copyMakeBorder(inputs[i],0,0,(h-w)//2,(h-w)//2,cv2.BORDER_CONSTANT)
    elif(w > h): inputs[i] = cv2.copyMakeBorder(inputs[i],(w-h)//2,(w-h)//2,0,0,cv2.BORDER_CONSTANT)
    inputs[i] = cv2.resize(inputs[i],(50,50))
    if(i%100 == 0): print(i) 
print(inputs.shape)
output = open('C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\grey_4_sq_inputs.pkl','wb')
pickle.dump(inputs, output)
output.close() """

"""with open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\realimageinputs.pkl",'rb') as pkl: inputs = pickle.load(pkl)
with open("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\ClassifierData\\coordlabels.pkl", 'rb') as pkl: labels = pickle.load(pkl)
inputs = np.array(inputs)
labels = np.array(labels) 

print(labels.shape)"""


"""for i in range(10):
    index = random.randint(0,len(inputs))
    chunk = inputs[index]
    cv2.imshow('img',chunk)
    cv2.waitKey()"""

"""x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

print(x_train.shape)

input_shape = (50,50,1)

model = Sequential(
    [
        Input(shape=input_shape),
        Conv2D(10,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(pool_size=(3,3)),
        Conv2D(30,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(pool_size=(3,3)),
        Dropout(0.3),
        Flatten(),
        Dense(16, activation = "relu"),
        Dense(4,activation="relu")
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
    metrics=['accuracy']
)

print("Model Compiled")

model.fit(x_train, y_train,epochs=3,validation_data=(x_val,y_val),shuffle=True)


results = model.evaluate(x_test,y_test)
print(f"Test Loss: {results[0]} \nTest Accuracy: {results[1]}")

model.save("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\models\\model_coord_fast")

print("Saved Model.") """
