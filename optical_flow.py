import cv2
import sys
import numpy as np 
#from matplotlib import pyplot as plt 
import os


frate=24
delay= int(1000/24)
path= "../Final_Dataset_6sec"
files=os.listdir(path)
#files=[ 'b0.05','b50','b100','b180','b0.0005','b40','b0.005','b450','b110','b0.5','b60','b0.1','b25','b1', 'b600']
#files=os.listdir(path)
#print(files)
video_file_path= "../"
model = ResNet50(weights='imagenet')
model_new = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
for i,folders in enumerate(files):
	folder_path=os.path.join(path,folders)
	class_files= os.listdir(folder_path)
	video_vector_folder= os.path.join(video_file_path,folders)
	if not os.path.exists(video_vector_folder):

		os.makedirs(video_vector_folder)


	cnt=0
	for j, videos in enumerate(class_files):
		video_path= os.path.join(folder_path,videos)
		vid_file=cv2.VideoCapture(video_path)
		lst=[]
		vid_name=os.path.splitext(videos)[0]
		while(vid_file.isOpened()):
			ret, frame = vid_file.read()
			if (ret==True):
				img=frame[100:650,300:950]
				img = cv2.resize(img,(224,224))
				lst.append(img)
				if (cv2.waitKey(25) & 0xFF == ord('q')):
					break
			else:
				break


		lst=np.asarray(lst)
		print(lst.shape,video_path)
		lst=model_new.predict(lst)
		#print(folders,videos,lst.shape)
		np.save(os.path.join(video_vector_folder, vid_name+".npy"),lst)
		cnt+=1





vid_file.release()

cv2.destroyAllWindows()