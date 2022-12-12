import tensorflow as tf
from tensorflow import keras
from keras import Sequential
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import genfromtxt
import csv
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import torch
import torchvision.ops.boxes
from torchvision.ops.boxes import _box_inter_union
import cv2
import time

# with open('c:\\users\\aaronpeng\\desktop\\PRSEF\\data\\t\\predictions-set00-V000.json', 'r') as f:
#   data = json.load(f)
# print(data)
model = Sequential()

#def cLoss(y_pred,y_true):


# for fnum in range(1711):
#     dir = f'c:/users/aaronpeng/desktop/prsef/data/set01/V000/Images/frame{fnum}.png'
#     pic = Image.open(dir)
#     arr = np.array(pic)
#     f = open('data/input.csv', 'w')
#     writer = csv.writer(f)
#     writer.writerow(arr)
#     f.close()

# data = genfromtxt('data/input.csv',delimiter=',')
# im = Image.fromarray(data)
# im.save("test.png")

json_file_path = "C:\\Users\\AaronPeng\\Desktop\\PRSEF\\data\\annotations\\predictions-set01-V000.json"

with open(json_file_path, 'r') as j:
     data = json.loads(j.read())

def showFrame(frameNum, cThresh=0.6, sThresh=1000):
    dims = (640,480)
    file = data[frameNum]['filename']
    annotations = data[frameNum]['objects']
    img = cv2.imread(file)
    smalls = []

    for i in range(len(annotations)):
        obj = annotations[i]
        w = int(obj["relative_coordinates"]['width']*dims[0])
        h = int(obj["relative_coordinates"]['height']*dims[1])
        print((w,h))
        if((w*h < 1000 or obj['confidence']<.6) and obj['class_id'] != 9): 
            smalls.insert(0,i)

    for i in smalls: del annotations[i]

    for obj in annotations:
        c_X = int(obj["relative_coordinates"]['center_x']*dims[0])
        c_Y = int(obj["relative_coordinates"]['center_y']*dims[1])
        w = int(obj["relative_coordinates"]['width']*dims[0])
        h = int(obj["relative_coordinates"]['height']*dims[1])
        cv2.rectangle(img,(c_X-int(w/2),c_Y-int(h/2)),(c_X+int(w/2),c_Y+int(h/2)),(0,255,0))

    cv2.imshow("images",img)
    cv2.waitKey()

for i in range(1711):
    showFrame(i)
    time.sleep(1.5)
    cv2.destroyAllWindows()

"""
model.add(Conv2D(input_shape=(640,480,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dense(units=5,activation='softmax'))
from keras.optimizers import Adam
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()

#Label format:
#[
# {
#   top: A
#   left: B
#   width: C 
#   height: D
# }
#]

def inter_area(s1, s2) -> int:
	bl_a_x, bl_a_y, tr_a_x, tr_a_y = s1[0], s1[1], s1[2], s1[3]
	bl_b_x, bl_b_y, tr_b_x, tr_b_y = s2[0], s2[1], s2[2], s2[3]
	
	return (
		(min(tr_a_x, tr_b_x) - max(bl_a_x, bl_b_x))
		* (min(tr_a_y, tr_b_y) - max(bl_a_y, bl_b_y))
	)

def giou_loss(input_boxes, target_boxes, eps=1e-7):
    inter = inter_area()


print(giou_loss([],[]))
"""