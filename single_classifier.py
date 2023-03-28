import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization,Flatten, Dropout
import pickle
from utils import squarifyPed, get_inputs
from sklearn.model_selection import train_test_split
import keras.datasets
import os
import matplotlib.pyplot as plt
from keras.losses import CategoricalCrossentropy
from keras.regularizers import L2

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
    if(len(realx) == 20000): break
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
for i in range(4800):
    for j in range(1,3):
        pimg = cv2.imread(f"{ped}\\{j}\\ped_examples\\img_{'0'*(5-len(str(i)))}{i}.pgm",0)
        npimg = cv2.imread(f"{ped}\\{j}\\non-ped_examples\\img_{'0'*(5-len(str(i)))}{i}.pgm",0)
        pimg = cv2.resize(pimg,(32,32))
        npimg = cv2.resize(npimg,(32,32))
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
output.close() 

"""

"""

background = []

for i in range(7):
    img = np.expand_dims(cv2.imread(f"{script_dir}\\ClassifierData\\test{i}.jpg",0),-1)
    chunks, dims = get_inputs(img,[(1,1)],[32],np.sum(img.shape)//150,(32,32))
    for chunk in chunks: background.append(chunk)

background = np.array(background)
print(background.shape)

output = open(f"{script_dir}\\classifierdata\\extrabck.pkl","wb")
pickle.dump(background,output)
output.close()
"""


def augment(images,labels):
    imgRet = []
    labelRet = []
    for i in range(len(images)):
        img = images[i]
        flipped = cv2.flip(img,1)
        dimmed = img/2
        imgRet.append(img)
        imgRet.append(flipped)
        imgRet.append(dimmed)
        imgRet.append(flipped/2)
        for j in range(4):
            labelRet.append(labels[i])
    return np.array(imgRet), np.array(labelRet)

#with open(f"{script_dir}\\ClassifierData\\cars.pkl",'rb') as pkl: carinputs = pickle.load(pkl)
with open(f"{script_dir}\\ClassifierData\\cifarX.pkl",'rb') as pkl: cifarinputs = pickle.load(pkl)
with open(f"{script_dir}\\ClassifierData\\cifarY.pkl",'rb') as pkl: cifarlabels = pickle.load(pkl)
#with open(f"{script_dir}\\ClassifierData\\truck.pkl",'rb') as pkl: truckinputs = pickle.load(pkl)
#with open(f"{script_dir}\\ClassifierData\\peds.pkl",'rb') as pkl: pedinputs = pickle.load(pkl)
with open(f"{script_dir}\\ClassifierData\\pedinputs.pkl",'rb') as pkl: pedinputs = pickle.load(pkl)
with open(f"{script_dir}\\ClassifierData\\extrabck.pkl",'rb') as pkl: extra_backgrounds = pickle.load(pkl)

#carinputs = np.array(carinputs)
#truckinputs = np.array(truckinputs)
cifarinputs = np.array(cifarinputs)
pedinputs = np.array(pedinputs)
backgrounds = np.array(extra_backgrounds)#[:50000]

#carinputs = augment(carinputs)[:15000]
cifarinputs, cifarlabels = augment(cifarinputs,cifarlabels)
backgrounds, blabels = augment(backgrounds,[[1,0,0,0]]*backgrounds.shape[0])
#truckinputs = augment(truckinputs)

#print(f" cars: {carinputs.shape} \n trucks: {truckinputs.shape} \n pedestrians: {pedinputs.shape} \n background: {backgrounds.shape}")
print(f" cars N trucks: {cifarinputs.shape} \n pedestrians: {pedinputs.shape} \n background: {backgrounds.shape}")


# inputs = np.concatenate(
#     (backgrounds,pedinputs,carinputs,truckinputs)
# )
inputs = np.concatenate(
    (backgrounds,pedinputs,cifarinputs)
)
inputs = inputs/255

import random
for i in range(5):
    num = random.randint(0,len(pedinputs))
    plt.imshow(pedinputs[num],cmap='gray')
    plt.show()


# labels = np.concatenate(
#     (
#         [[1,0,0,0]]*backgrounds.shape[0],
#         [[0,1,0,0]]*pedinputs.shape[0],
#         [[0,0,1,0]]*carinputs.shape[0],
#         [[0,0,0,1]]*truckinputs.shape[0]
#     )
# )
labels = np.concatenate(
    (
        blabels,
        [[0,1,0,0]]*pedinputs.shape[0],
        cifarlabels
    )
)

del cifarinputs
del cifarlabels
#del carinputs
del pedinputs
#del truckinputs
del backgrounds

print(f"inputs: {inputs.shape}")
print(f"labels: {labels.shape}")

x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

# import random
# for i in range(20):
#      index = random.randint(0,len(x_train)-1)
#      print(y_train[index])
#      plt.imshow(x_train[index],cmap='gray')
#      plt.show()

input_shape = (32,32,1)

model = Sequential(
    [
        Input(shape=input_shape),
        Conv2D(6,kernel_size=(5,5),activation='relu'),
        AveragePooling2D(2,2),
        Conv2D(16,kernel_size=(5,5),activation='relu'),
        AveragePooling2D(2,2),
        Flatten(),
        Dense(120, activation = 'relu'),
        Dense(84,activation='relu'),
        Dense(4,activation="softmax")
    ]
)    
    


callback = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,
    restore_best_weights = True,
    verbose = 1,
    start_from_epoch = 1
)

loss = CategoricalCrossentropy(label_smoothing=0.15)
opt = keras.optimizers.Adam(learning_rate = 1e-3, decay = 1e-5)

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy']
)

print("Model Compiled")

#print(model.summary())



history = model.fit(
    x_train,y_train,
    epochs = 100,
    validation_data = (x_val,y_val),
    callbacks = [callback],
    shuffle = True
)

model.fit()

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

exit()

print(f"Test Loss: {results[0]} \nTest Accuracy: {results[1]}")
model.save(f"{script_dir}\\32x32models\\ensemble_1\\m4")
print("Saved Model.") 