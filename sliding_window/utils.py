import cv2
import numpy as np
import glob
import pickle
from ensemble_boxes import *

def drawOutputs(file,cls,coords):
    img = cv2.imread(file)
    cv2.rectangle(img,(10,10),(100,100),(0,255,0))
    cv2.putText(img, "box 1", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
    return img

def load_data(num, grayscale = False):
    with open(f"C:\\Users\\AaronPeng\\Desktop\\PRSEF\\sliding_window\\pkls\\img{(num+1)*100}{'gs' if grayscale else ''}.pkl",'rb') as pkl:
        res = pickle.load(pkl)
        if(grayscale): res = np.expand_dims(res,-1)
        return res

def single_img(dir, grayscale):
    if(grayscale): img = np.expand_dims(cv2.imread(dir, 0),-1)
    else: img = cv2.imread(dir)
    img = squarify(img, (50,50))
    if(grayscale):
        img = np.expand_dims(img,-1)
    return img, img.shape

def squarify(chunk, final_dims):
    h,w,c = chunk.shape
    if(h > w): chunk = cv2.copyMakeBorder(chunk,0,0,(h-w)//2,(h-w)//2,cv2.BORDER_CONSTANT)
    elif(w > h): chunk = cv2.copyMakeBorder(chunk,(w-h)//2,(w-h)//2,0,0,cv2.BORDER_CONSTANT)
    chunk = cv2.resize(chunk,final_dims)
    return chunk

def get_inputs(img,ratios,sizes,stride,final_dims):
    ret = []
    dims = []
    imgW,imgH,c = img.shape
    for ratio in ratios:
        for size in sizes:
            width = int(ratio[0] * size)
            height = int(ratio[1] * size)
            for row in range((imgH-height)//stride):
                for col in range((imgW-width)//stride):
                    chunk = img[stride*col:stride*col+height,stride*row:stride*row+width]
                    dims.append((stride*col/300,stride*row/480,width/300, height/480))
                    ret.append(squarify(chunk,final_dims))
    return np.array(ret),dims

def NMS(boxes, scores, labels):
	return nms([boxes],[scores],[labels],weights=None)

def get_results(res):
    classes = {
    '"pedestrian"' : 0,
    '"biker"' : 1,
    '"car"' : 2,
    '"truck"' : 3,
    '"trafficLight"' : 4,
    }
    ret = []
    scores = []
    for score in res:
        max = -1
        index = 0
        for key in classes.keys():
            if(score[classes[key]] > max):
                max = score[classes[key]]
                index = classes[key]
        if(max < 0.5): 
            index = 5
            scores.append(1-max)
        else: scores.append(max)
        ret.append(index)
    return ret, scores