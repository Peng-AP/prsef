from classifier import single_predictor
from utils import load_data, get_inputs, get_results, NMS, squarify, drawOutputs
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGE_DIMENSIONS = (200,200)
RATIOS = [(1,1),(1,1.5),(1.5,1)]
SIZES = [30,50,75,100,150]
    
classifier = single_predictor("cifar_ped_model_final")

img = np.expand_dims(cv2.imread("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\sliding_window\\imgs\\red_car.png",0),-1)/255
print(img.shape)
batch = [img]

for image in batch:
     inputs,dims = get_inputs(image,RATIOS,SIZES,5, (32,32), IMAGE_DIMENSIONS)
     results = classifier.predict(inputs)    
     print(results.shape,dims.shape)
     scores = [[],[],[],[]]
     for res in results:
          max = -1
          index = 0
          for i in range(4):
               if(res[i] > max):
                    max = res[i]
                    index = i
          if(max > 0.4): scores[index].append(max)
     new_img = cv2.resize(img,IMAGE_DIMENSIONS)
     for j in range(1,4):
          filtered,conf = NMS(dims,np.array(scores[j]),0.5)
          print(filtered.shape)
          print(filtered,conf)
          boxes = []
          for i in range(len(filtered)-2,len(filtered)):
               boxes.append([filtered[i][0],filtered[i][1],filtered[i][2],filtered[i][3],conf[i]])
          print(boxes)
          new_img = drawOutputs(new_img,boxes,("pedestrian" if j==1 else ("car" if j==2 else "truck")))
     plt.imshow(new_img,cmap='gray')
     plt.show()


    
