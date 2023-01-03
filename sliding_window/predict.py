from classifier import single_predictor
from utils import load_data, get_inputs, get_results, NMS
import time
import numpy as np
import cv2

IMAGE_DIMENSIONS = (480,300)
RATIOS = [(1,1),(1,1.5),(1.5,1)]
SIZES = [10,25,50]
    
classifier = single_predictor("model_coord_4")
batch = load_data(23,True)

""" 
     """
for image in batch:
    inputs,dims = get_inputs(image,RATIOS,SIZES,20, (50,50))
    results = classifier.predict(inputs)    
    print(results[0])
    cv2.imshow('',inputs[0])
    cv2.waitKey()
    
