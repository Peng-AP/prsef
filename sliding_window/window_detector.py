import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import glob
from classifier import single_predictor
from utils import squarify, single_img, drawOutputs
import cv2
import numpy as np

script_dir = os.path.dirname(__file__)
grayScale = False

def sel_img():
    cur_img = filedialog.askopenfilename(initialdir=script_dir + '\\imgs', title="Select Demo Image", filetypes=[("all files", "*.*"), ("png files","*.png"), ("jpg files", "*.jpg"), ("jpeg files", "*.jpeg")])
    newimg = ImageTk.PhotoImage(Image.open(cur_img))
    demoImg.configure(image = newimg)
    demoImg.image = newimg

def predict():
    print(cur_img)
    classifier_model = single_predictor(classModel.get())
    coordinator_model = single_predictor(coordModel.get())
    img,shape = single_img(cur_img, True)
    img = np.expand_dims(img,0)
    cls = classifier_model.predictor.predict(img)
    coords = coordinator_model.predictor.predict(img)
    disp = cv2.cvtColor(drawOutputs(f"{script_dir}\\imgs\\blue_car.jpg",cls,coords),cv2.COLOR_BGR2RGB)
    new_img = ImageTk.PhotoImage(image=Image.fromarray(disp))
    demoImg.configure(image = new_img)
    demoImg.image = new_img

root = Tk(screenName="Sliding-Window Object Detector")
root.geometry("600x500")

image_frame = Frame(root, padx=10, pady=10, highlightbackground='black', highlightthickness=3)
image_frame.grid(column = 5,row = 1, padx = 10, pady=10)

cur_img = f"{script_dir}\\imgs\\DummyImage.png"
img = ImageTk.PhotoImage(Image.open(cur_img))
demoImg = Label(image_frame, text="img", image=img)
demoImg.pack()

options = Frame(root, padx = 5, pady = 5)
options.grid(column = 5, row = 0)

file_btn = Button(options, text = "Choose Image File", padx=5, pady = 2, command = sel_img)
file_btn.grid(column = 0, row = 0, padx = 5)

analyze_btn = Button(options, text = "Predict Bounding Boxes", padx = 5, pady = 2, command = predict)
analyze_btn.grid(column = 1, row = 0, padx = 5)

selectors = Frame(root, padx = 5, pady = 5)
selectors.grid(column = 6, row = 1)

classModel = StringVar()
coordModel = StringVar()

classModelSelection = OptionMenu(selectors, classModel, '')
coordModelSelection = OptionMenu(selectors, coordModel, '')

classLabel = Label(selectors, text = "Classifier:")
coordLabel = Label(selectors, text = "Coordinate Predictor:")

for model in glob.glob(script_dir + "/models/*"):
    model = model[len(script_dir) + 8:]
    if("coord" not in model and "Desc" not in model):
        classModelSelection['menu'].add_command(label=model, command=tk._setit(classModel, model))
    elif("coord" in model):
        coordModelSelection['menu'].add_command(label=model, command=tk._setit(coordModel, model))

classLabel.grid(column = 0, row = 0)
classModelSelection.grid(column = 0, row = 1)
coordLabel.grid(column = 0, row = 2)
coordModelSelection.grid(column = 0, row = 3)


root.mainloop()

