from classifier import single_predictor
from utils import load_data, get_inputs, get_results, NMS, squarify, drawOutputs
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

dims = [
    80, #topX
    175, #topY
    102, #width
    102  #height
]

model = single_predictor("small_4","72x72")

image = cv2.imread("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\sliding_window\\imgs\\standard1.jpg",0)

chunk = image[dims[1]:dims[1]+dims[3],dims[0]:dims[0]+dims[2]]

plt.imshow(chunk,cmap='gray')
plt.show()

testImg = np.expand_dims(cv2.resize(chunk,(72,72)),0)/255

plt.imshow(cv2.resize(chunk,(72,72)),cmap='gray')
plt.show()

print(testImg.shape)

print(model.predictor.predict([testImg]))