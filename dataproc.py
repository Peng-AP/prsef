import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle

def read_all_images(path_to_data):
    ret = []
    for i in range(15660):
        path = f"\\pos{'0'*(5-len(str(i)))}{i}.pgm"
        img = cv2.imread(path_to_data+path,0)
        img = cv2.resize(img,(72,72))
        ret.append(img)
    return ret



def grayIMG(imgs):
    ret = []
    for i in range(len(imgs)):
        ret.append(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY))
    return np.array(ret)

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def trucks(imgs,lbls):
    ri = []
    for i in range(len(imgs)):
        if(lbls[i] == 10):
            ri.append(cv2.resize(imgs[i],(72,72)))
    return np.array(ri)

imgs = read_all_images("C:\\Users\\AaronPeng\\Desktop\\TrainingData.tar\\DaimlerBenchmark\\Data\\TrainingData\\Pedestrians\\48x96")
script_dir = os.path.dirname(__file__)


plt.imshow(imgs[175],cmap='gray')
plt.show()

output = open(f"{script_dir}\\ClassifierData\\peds.pkl",'wb')
pickle.dump(imgs,output)
output.close()