import cv2
import sys
import numpy as np 
#from matplotlib import pyplot as plt 
import os
import tensorflow
import keras



 ## Creats dataset of images using cropped images

 # Outputs a 4d array with dimensions(number of images, image dimensions)

 ## Don't mess up with individual lines of code.

 ## feel free to change path names and texture dictionaries in case we render more videos
def create_data():
	path= 'CroppedImages'
	files=os.listdir(path)
	print(files)
	cnt=0
	# length=5500
	# label=np.zeros((length,))
	#input_data=np.zeros((length,250,250,3))

	diction= {"Bro":0, "Cotton":1,"FeltGreen": 2, "Sha":3,"Silk":4, "Sweater":5, "Fleece":6}
	a=[]
	b=[]
	c=[]


	count=0
	for label_idx, i in enumerate(files):
		file_path=os.path.join(path,i)
		new_files=os.listdir(file_path)


		for idx, j in enumerate(new_files):
			img=cv2.imread(os.path.join(file_path,j))


			img_name=os.path.splitext(j)[0]
			spt=img_name.split("_")
			spt[2]=spt[2][1:]
			spt[3]=spt[3][1:]
			

			a.append(img)
			b.append(float(spt[2]))
			c.append(diction[spt[0]])
			print(float(spt[2]),diction[spt[0]],spt[0],count)


			# if(count%1000==0):

			# 	print(img.shape, count,i,j)

			count+=1
	input_data=np.asarray(a)
	bs_label=np.asarray(b)
	texture_label=np.asarray(c)


	#input_data=input_data[0:count]		
	input_data = input_data.reshape(input_data.shape[0], 3, 250,250)
	#input_data = input_data.astype('uint8')
	# input_data /= 255


	print(bs_label.shape,texture_label.shape,input_data.shape)
	np.save('inputs.npy',input_data)
	np.save('bs_label.npy',bs_label)
	np.save('texture_label.npy',texture_label)
	


## Call the function
create_data()

# input_data=np.load('inputs.npy')
# label=np.load('label.npy')
# final_label=keras.utils.to_categorical(label,num_classes=40)
# print(label.shape,final_label[8500])