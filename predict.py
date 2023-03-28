from classifier import single_predictor
from utils import load_data, get_inputs, get_results, NMS, squarify, drawOutputs
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import glob
import copy

RATIOS = [(1,1),(1,1.5),(1.5,1)]
SIZES = np.array([30,50,75,100,120])
INPUTSIZE = (32,32)
    
classifiers = ["LeNet_Drop_LRS_ActReg"]
#classifiers = glob.glob("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\sliding_window\\32x32models\\ensemble_1\\*")

img_dir = "C:\\Users\\AaronPeng\\Desktop\\PRSEF\\sliding_window\\imgs\\standard2.jpg"

colorImg = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)

heatImg = copy.deepcopy(colorImg)

W,H = colorImg.shape[0:2]

alpha = 1
overlay = heatImg.copy()
cv2.rectangle(overlay,(0,0),(colorImg.shape[1],colorImg.shape[0]),(0,0,0),-1)
heatImg = cv2.addWeighted(overlay, alpha, heatImg, 1 - alpha, 0)


image = cv2.imread(img_dir,0)

img = np.expand_dims(image,-1)/255

scale = [img.shape[0]/250,img.shape[1]/250]
SIZES = np.floor(SIZES*np.average(scale))
print(SIZES)

plt.imshow(colorImg)
plt.show()

models = [
     "LeNet_smooth_L2",
]

#plt.imshow(image_sharp,cmap="gray")
#plt.show()
batch = [img]
imgs = []
for model in models:
     for image in batch:
          colorImg = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
          inputs,dims = get_inputs(image,RATIOS,SIZES,10, INPUTSIZE)

          print(inputs.shape)
          results = np.average([(single_predictor(model,"32x32").predict(inputs)) for c in classifiers],0)    
          print(results.shape,dims.shape)
          scores = [[],[],[],[]] #backgrounds,peds,cars,trucks
          dimensions = [[],[],[],[]]

          #split into classes
          for i in range(results.shape[0]):
               res = results[i]
               dim = dims[i]
               sorted = np.argsort(res)
               max = res[sorted[-1]]
               if(max > 0.5): 
                    scores[sorted[-1]].append(max)
                    dimensions[sorted[-1]].append(dim)

          #scores[2] = np.concatenate((scores[2],scores[3]))
          #dimensions[2] = np.concatenate((dimensions[2],dimensions[3]))

          #nms and draw
          for j in range(1,4):
               if(len(scores[j])>0):
                    filtered,conf = NMS(np.array(dimensions[j]),np.array(scores[j]),0.4)
                    print(filtered.shape)
                    boxes = []
                    for i in range(filtered.shape[0]//5):
                         boxes.append([filtered[i][0],filtered[i][1],filtered[i][2],filtered[i][3],conf[i]])
                    print(boxes, j)
                    colorImg,heatImg = drawOutputs(colorImg,boxes,("ped" if j==1 else ("car" if j==2 else "truck")),heatImg)
          imgs.append(colorImg.copy())
for i in imgs:
     plt.imshow(i)
     plt.show()
