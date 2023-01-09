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
from utils import squarifyPed
from sklearn.model_selection import train_test_split
import math
import keras.datasets
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

script_dir = os.path.dirname(__file__)
user = 'AaronPeng'

"""Data"""
num_classes = 5
classes = {
    'background' : 0,
    'pedestrian' : 1,
    'automobile' : 2,
    'truck' : 3
}
ped = "C:\\Users\\AaronPeng\\Downloads\\DC-ped-dataset_base.tar"
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

realx = []
realy = []
nt = 0
for i in range(50000):
    if(len(realx) == 2000): break
    if(y_train[i] == 1):
        realx.append(x_train[i][:,:,0])
        realy.append([0,0,1,0])
    elif(y_train[i] == 9):
        nt += 1
        realx.append(x_train[i][:,:,0])
        realy.append([0,0,0,1])

print(nt)
print(".")
peds = []
npeds = []
for i in range(1000):
    pimg = cv2.imread(f"{ped}\\1\\ped_examples\\img_{'0'*(5-len(str(i)))}{i}.pgm",0)
    npimg = cv2.imread(f"{ped}\\1\\non-ped_examples\\img_{'0'*(5-len(str(i)))}{i}.pgm",0)
    peds.append(pimg)
    npeds.append(npimg)


output = open(f"{script_dir}\\ClassifierData\\pedinputs.pkl",'wb')
pickle.dump(peds,output)
output.close() 

output = open(f"{script_dir}\\ClassifierData\\npedinputs.pkl",'wb')
pickle.dump(npeds,output)
output.close()

output = open(f"{script_dir}\\ClassifierData\\cifarX.pkl",'wb')
pickle.dump(realx,output)
output.close()

output = open(f"{script_dir}\\ClassifierData\\cifarY.pkl",'wb')
pickle.dump(realy,output)
output.close()"""
"""


anns = pd.read_csv("C:\\Users\\xaep\\Desktop\\PRSEF\\labels.csv").to_numpy()
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
output = open("C:\\Users\\xaep\\Desktop\\PRSEF\\ClassifierData\\realimageinputs.pkl",'wb')
pickle.dump(inputs,output)
output.close() 
output = open("C:\\Users\\xaep\\Desktop\\PRSEF\\ClassifierData\\coordlabels.pkl",'wb')
pickle.dump(coordinateLabels,output)
output.close()



#for dir in glob.glob(image_dirs):
    #img = cv2.resize(cv2.imread(dir),(192,120))
    #print(img.shape)
    #cv2.imshow("img",img)
    #cv2.waitKey()

with open("C:\\Users\\xaep\\Desktop\\PRSEF\\ClassifierData\\grey_4inputs.pkl",'rb') as pkl: inputs = pickle.load(pkl)
for i in range(len(inputs)):
    h,w,c = inputs[i].shape
    if(h > w): inputs[i] = cv2.copyMakeBorder(inputs[i],0,0,(h-w)//2,(h-w)//2,cv2.BORDER_CONSTANT)
    elif(w > h): inputs[i] = cv2.copyMakeBorder(inputs[i],(w-h)//2,(w-h)//2,0,0,cv2.BORDER_CONSTANT)
    inputs[i] = cv2.resize(inputs[i],(50,50))
    if(i%100 == 0): print(i) 
print(inputs.shape)
output = open('C:\\Users\\xaep\\Desktop\\PRSEF\\ClassifierData\\grey_4_sq_inputs.pkl','wb')
pickle.dump(inputs, output)
output.close() """

with open(f"{script_dir}\\ClassifierData\\cifarX.pkl",'rb') as pkl: cifarinputs = pickle.load(pkl)
with open(f"{script_dir}\\ClassifierData\\cifarY.pkl",'rb') as pkl: cifarlabels = pickle.load(pkl)
with open(f"{script_dir}\\ClassifierData\\npedinputs.pkl", 'rb') as pkl: npedinputs = pickle.load(pkl)
with open(f"{script_dir}\\ClassifierData\\pedinputs.pkl",'rb') as pkl: pedinputs = pickle.load(pkl)

for i in range(1000):
    pedinputs[i] = squarifyPed(pedinputs[i],npedinputs[999-i])
    npedinputs[i] = squarifyPed(npedinputs[i],npedinputs[999-i])
for i in range(1000):
    pedinputs[i],npedinputs[i] = cv2.resize(pedinputs[i],(32,32)),cv2.resize(npedinputs[i],(32,32))
pedinputs = np.array(pedinputs)
npedinputs = np.array(npedinputs)
print(np.array(cifarinputs).shape,pedinputs.shape,npedinputs.shape)
inputs = np.concatenate((cifarinputs,npedinputs,pedinputs),0)
labels = np.concatenate((cifarlabels,[[1,0,0,0]]*1000,[[0,1,0,0]]*1000),0) 

inputs = inputs/255

print(inputs.shape)


x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

augx_train = []
augy_train = []

for i in range(len(x_train)):
    img = x_train[i]
    flipped = cv2.flip(img,1)
    dimmed = img/2
    augx_train.append(img)
    augx_train.append(flipped)
    augx_train.append(dimmed)
    augx_train.append(flipped/2)
    """
    plt.imshow(img,cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.imshow(flipped,cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.imshow(dimmed,cmap='gray', vmin=0, vmax=1)
    plt.show()
    """
    for i in range(4): augy_train.append(y_train[i])

augx_train,augy_train = np.array(augx_train),np.array(augy_train)

print(augx_train.shape)


input_shape = (32,32,1)

model = Sequential(
    [
        Input(shape=input_shape),
        Conv2D(2,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(4,kernel_size=(3,3),activation='relu'),
        Conv2D(6,kernel_size=(3,3),activation='relu'),
        Dropout(0.3),
        Flatten(),
        Dense(10, activation = "relu"),
        Dense(4,activation="softmax")
    ]
)

callback = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=15,
    restore_best_weights = True,
    verbose = 1,
    start_from_epoch = 50
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled")

history = model.fit(
    x_train,y_train,
    epochs = 100,
    validation_data = (x_val,y_val),
    callbacks = [callback],
    shuffle = True
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

results = model.evaluate(x_test,y_test)
print(f"Test Loss: {results[0]} \nTest Accuracy: {results[1]}")

if(results[1] > .8533):
    model.save(f"{script_dir}\\models\\cifar_ped_model_final")
    print("Saved Model.") 