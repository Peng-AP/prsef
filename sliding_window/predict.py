from classifier import single_predictor
from utils import load_data, get_inputs, get_results, NMS, squarify, drawOutputs
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

RATIOS = [(1,1),(1,1.5),(1.5,1)]
SIZES = np.array([30,50,75,100,150])
    
classifier = single_predictor("LeNet_background_smooth")

img_dir = "C:\\Users\\AaronPeng\\Desktop\\PRSEF\\sliding_window\\imgs\\standard2.jpg"
colorImg = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)

image = cv2.imread(img_dir,0)

img = np.expand_dims(image,-1)/255

scale = [img.shape[0]/250,img.shape[1]/250]
SIZES = np.floor(SIZES*np.average(scale))
print(SIZES)

plt.imshow(colorImg)
plt.show()
#plt.imshow(image_sharp,cmap="gray")
#plt.show()
batch = [img]

for image in batch:
     inputs,dims = get_inputs(image,RATIOS,SIZES,5, (32,32))

     print(inputs.shape)
     results = classifier.predict(inputs)    
     print(results.shape,dims.shape)
     scores = [[],[],[],[]] #backgrounds,peds,cars,trucks
     dimensions = [[],[],[],[]]

     #split into classes
     for i in range(results.shape[0]):
          res = results[i]
          dim = dims[i]
          sorted = np.argsort(res) #find highest score
          max = res[sorted[-1]]
          #if(dim[0] == 25 and dim[2] == 140):
               #print(res)
               #plt.imshow(inputs[i])
               #plt.show()
          if(max > 0.5): 
               scores[sorted[-1]].append(max)
               dimensions[sorted[-1]].append(dim)

     #nms and draw
     for j in range(1,4):
          filtered,conf = NMS(np.array(dimensions[j]),np.array(scores[j]),0.4)
          print(filtered.shape)
          boxes = []
          for i in range(filtered.shape[0]//5 + 1):
               boxes.append([filtered[i][0],filtered[i][1],filtered[i][2],filtered[i][3],conf[i]])
          print(boxes, j)
          new_img = drawOutputs(colorImg,boxes,("ped" if j==1 else ("car" if j==2 else "truck")))
     plt.imshow(colorImg)
     plt.show()


    
