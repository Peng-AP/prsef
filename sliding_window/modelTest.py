from classifier import single_predictor
from utils import load_data, get_inputs, get_results, NMS, squarify, drawOutputs
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

dims = [
    343, #topX
    106, #topY
    247, #width
    146  #height
]

model = single_predictor("lenet_background_aug")

image = cv2.imread("c:\\Users\\AaronPeng\\desktop\\prsef\\sliding_window\\imgs\\road.png",0)

chunk = image[dims[1]:dims[1]+dims[3],dims[0]:dims[0]+dims[2]]

plt.imshow(chunk,cmap='gray')
plt.show()

testImg = np.expand_dims(cv2.resize(chunk,(32,32)),0)/255

print(testImg.shape)

print(model.predictor.predict([testImg]))