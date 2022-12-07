import tensorflow as tf
from tensorflow import keras
from keras import Sequential
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import genfromtxt
import csv

#hello

# with open('c:\\users\\aaronpeng\\desktop\\PRSEF\\data\\t\\predictions-set00-V000.json', 'r') as f:
#   data = json.load(f)
# print(data)
model = Sequential()

for fnum in range(1711):
    dir = f'c:/users/aaronpeng/desktop/prsef/data/set01/V000/Images/frame{fnum}.png'
    pic = Image.open(dir)
    arr = np.array(pic)
    f = open('data/input.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(arr)
    f.close()

data = genfromtxt('data/input.csv',delimiter=',')
im = Image.fromarray(data)
im.save("test.png")


