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

## extract frames from  video.

## I select 130 frames out of 150 as first 10 and last 10 are useless for me.

## Change if you dont like it
frate=24
delay= int(1000/24)
frame_list= [i for i in range(10,140)]
if not os.path.exists("StiffImages"):
	os.makedirs("StiffImages")
#path="../"
#print(os.listdir(path))

path= "../Final_Dataset_6sec"
imagepath= "StiffImages"
files=os.listdir(path)
print(files)



for i,folders in enumerate(files):
	folder_path=os.path.join(path,folders)
	class_files= os.listdir(folder_path)
	image_folder_path= os.path.join(imagepath,folders)


	if not os.path.exists(image_folder_path):
		os.makedirs(image_folder_path)
	cnt=0


	for j, videos in enumerate(class_files):

		video_path= os.path.join(folder_path,videos)
		vid_file=cv2.VideoCapture(video_path)
		count=0
		print(folders,videos)

		vid_name_temp=os.path.splitext(videos)[0]


		while(vid_file.isOpened()):
			ret, frame = vid_file.read()
			if (ret==True):
				#print(ret, frame.shape,count)
				#cv2.imshow("Frame",frame)
				if count in frame_list:
					cnt+=1
					cv2.imwrite(os.path.join(image_folder_path,vid_name_temp+"_"+"frame%d.jpg"% cnt ),frame)

				if (cv2.waitKey(25) & 0xFF == ord('q')):
					break
			else:
				break

			count+=1





vid_file.release()

cv2.destroyAllWindows()






 