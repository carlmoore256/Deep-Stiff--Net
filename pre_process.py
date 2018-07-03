import cv2
import sys
import numpy as np 
#from matplotlib import pyplot as plt 
import os
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import plot_model



## takes image as input
## Crops  and resizes the image
if not os.path.exists("CroppedImages"):
	os.makedirs("CroppedImages")


path= "StiffImages"
crop_path= "CroppedImages"
files=os.listdir(path)
print(files)


cnt=0
for i,folders in enumerate(files):
	folder_path=os.path.join(path,folders)
	class_files= os.listdir(folder_path)
	image_folder_path= os.path.join(crop_path,folders)


	if not os.path.exists(image_folder_path):
		os.makedirs(image_folder_path)

	
	for j, images in enumerate(class_files):
		img = cv2.imread(os.path.join(folder_path,images))
		img=img[100:650,300:950]
		img = cv2.resize(img,(250,250))
		cv2.imwrite(os.path.join(image_folder_path,images),img)
		if(cnt%500==0):
			print(cnt)
		cnt+=1






