import os
import subprocess 

#dir = "C:\\Users\\AaronPeng\\Desktop\\PRSEF\\darknet-master\\"
#subprocess.run(["cd",dir])
#call = subprocess.run([f"{dir}darknet.exe","detect", f"{dir}cfg\\yolov4.cfg","yolov4.weights","-dont_show","-ext_output","data\\dog.jpg"],
#    stdout=subprocess.PIPE, text=True, cwd=dir)
#a = subprocess.run([f"{dir}darknet.exe","detect",f"{dir}cfg\\yolov3.cfg","yolov3.weights", "-ext_output"], cwd = dir)
#b = subprocess.run(["data\\dog.jpg"], stdout=subprocess.PIPE, text=True, cwd = dir)
#print(a.stdout)

# file = open('C:\\Users\\AaronPeng\\Desktop\\PRSEF\\darknet-master\\data\\images.txt','w')
# for i in range(1711):
#     file.write(f"C:\\Users\\AaronPeng\\Desktop\\PRSEF\\data\\set01\\V000\\images\\frame{i}.png\n")
# file.close() 

import cv2
import imutils
img = cv2.imread("C:\\Users\\AaronPeng\\Desktop\\PRSEF\\data\\set01\\V000\\images\\frame0.png")
img = imutils.resize(img, width=320)
cv2.imshow("img",img)
cv2.waitKey()

""" for i in range(1711):
    img = cv2.imread(f"C:\\Users\\AaronPeng\\Desktop\\PRSEF\\data\\set01\\V000\\images\\frame{i}.png")

    cv2.resize(img,None,fx=0.5,fy=0.5)

 """

