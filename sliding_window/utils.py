import cv2
import numpy as np
import random
import glob
import pickle
from ensemble_boxes import *

def squarifyPed(ped, nped):
    uno = random.randint(0,4)
    dos = random.randint(5,9)
    chunk1 = nped[0:36,uno:uno+9]
    chunk2 = nped[0:36,dos:dos+9]
    ped = np.concatenate((chunk1,ped,chunk2), axis = 1)
    return ped

def determineClass(cls):
    classes = {
        'background' : 0,
        'pedestrian' : 1,
        'automobile' : 2,
        'truck' : 3
    }
    conf = 0
    label = ''
    for key in classes.keys():
        if(cls[classes[key]] > conf):
            conf = cls[classes[key]]
            label = key
    return conf,label


def drawOutputs(img,boxes,name):
    classes = {
        'background' : 0,
        'pedestrian' : 1,
        'automobile' : 2,
        'truck' : 3
    }
    for box in boxes:
        x,y,w,h = box[:4]
        conf = box[4]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0))
        cv2.putText(img, f"{name}, {round(conf*100,1)}%", (x,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1,1,1), 1)
    return img

def load_data(num, grayscale = False):
    with open(f"C:\\Users\\xaep\\Desktop\\PRSEF\\sliding_window\\pkls\\img{(num+1)*100}{'gs' if grayscale else ''}.pkl",'rb') as pkl:
        res = pickle.load(pkl)
        if(grayscale): res = np.expand_dims(res,-1)
        return res

def single_img(dir, grayscale, shape):
    if(grayscale): img = np.expand_dims(cv2.imread(dir, 0),-1)
    else: img = cv2.imread(dir)
    img = squarify(img, shape)
    if(grayscale):
        img = np.expand_dims(img,-1)
    return img, img.shape

def squarify(chunk, final_dims):
    h,w,c = chunk.shape
    if(h > w): chunk = cv2.copyMakeBorder(chunk,0,0,(h-w)//2,(h-w)//2,cv2.BORDER_CONSTANT)
    elif(w > h): chunk = cv2.copyMakeBorder(chunk,(w-h)//2,(w-h)//2,0,0,cv2.BORDER_CONSTANT)
    chunk = cv2.resize(chunk,final_dims)
    return chunk

def get_inputs(img,ratios,sizes,stride,final_dims,img_dims):
    img = np.expand_dims(cv2.resize(img,img_dims),-1)
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
                    dims.append((stride*col,stride*row,width, height))
                    ret.append(squarify(chunk,final_dims))
    return np.array(ret),np.array(dims)

def NMS(boxes, scores, overlapThresh):
    if(len(boxes) == 0): return []
    if(boxes.dtype.kind == "i"):
        boxes = boxes.astype("float")
    picked = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1+boxes[:,2]
    y2 = y1+boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    while(len(idxs) > 0):
        last = len(idxs) - 1
        i = idxs[last]
        picked.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))

    return boxes[picked].astype("int"),scores[picked]

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