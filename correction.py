# import cv2
# import sys
# import numpy as np 
# #from matplotlib import pyplot as plt 
# import os
# import h5py
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from sklearn.model_selection import train_test_split
# from keras.utils import plot_model


# frate=24
# delay= int(1000/24)
# frame_list= [i for i in range(10,140)]
# if not os.path.exists("StiffImages"):
# 	os.makedirs("StiffImages")
# #path="../"
# #print(os.listdir(path))

# path= "Corrupt"
# imagepath1= "StiffImages"
# imagepath2="CroppedImages"
# files=os.listdir(path)
# print(files)



# for i,folders in enumerate(files):
# 	folder_path=os.path.join(path,folders)
# 	class_files= os.listdir(folder_path)
# 	image_folder_path_1= os.path.join(imagepath1,folders)
# 	image_folder_path_2= os.path.join(imagepath2,folders)


# 	if not os.path.exists(image_folder_path_1):
# 		os.makedirs(image_folder_path_1)
# 	cnt=0


# 	for j, videos in enumerate(class_files):

# 		video_path= os.path.join(folder_path,videos)
# 		vid_file=cv2.VideoCapture(video_path)
# 		count=0
# 		print(folders,videos)

# 		vid_name_temp=os.path.splitext(videos)[0]


# 		while(vid_file.isOpened()):
# 			ret, frame = vid_file.read()
# 			if (ret==True):
# 				#print(ret, frame.shape,count)
# 				#cv2.imshow("Frame",frame)
# 				if count in frame_list:
# 					cnt+=1
# 					cv2.imwrite(os.path.join(image_folder_path_1,vid_name_temp+"_"+"frame%d.jpg"% cnt ),frame)
# 					frame=frame[100:650,300:950]
# 					frame = cv2.resize(frame,(250,250))
# 					cv2.imwrite(os.path.join(image_folder_path_2,vid_name_temp+"_"+"frame%d.jpg"% cnt ),frame)

# 				if (cv2.waitKey(25) & 0xFF == ord('q')):
# 					break
# 			else:
# 				break

# 			count+=1





# vid_file.release()

# cv2.destroyAllWindows()






#  
## Written to correct faults in renderings

import cv2
import sys
import numpy as np 
#from matplotlib import pyplot as plt 
import os

import time
start_time = time.time()


width=1280
height=720


path="temp"

files=os.listdir(path)

save_path= "../Final_Dataset_6sec"

if not os.path.exists(save_path):
	os.makedirs(save_path)

for i, folders in enumerate(files):
	folder_path = os.path.join(path,folders)
	folder_save_path=os.path.join(save_path,folders)


	if not os.path.exists(folder_save_path):
		os.makedirs(folder_save_path)

	videos_inside= os.listdir(folder_path)

	for j,videos in enumerate(videos_inside):

		video_path=os.path.join(folder_path,videos)
		vid_name=os.path.splitext(videos)[0]
		vid_file   = cv2.VideoCapture(video_path)


		output1= os.path.join(folder_save_path, vid_name+"_"+str(1)+".avi")
		output2= os.path.join(folder_save_path, vid_name+"_"+str(2)+".avi")


		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out1 = cv2.VideoWriter(output1, fourcc, 25.0, (width, height))
		out2 = cv2.VideoWriter(output2, fourcc, 25.0, (width, height))


		count=0

		while(vid_file.isOpened()):
			ret2, frame2 = vid_file.read()

			if (ret2==True):
				#print(frame2.shape, count)


				if(count<150):
					out1.write(frame2)
					out2.write(frame2)

				count+=1



				if (cv2.waitKey(25) & 0xFF == ord('q')):
					break
			else:
				break

		print("--- %s seconds ---" % (time.time() - start_time))

# out1.release()
# out2.release()
vid_file.release()

cv2.destroyAllWindows()







	




count=0

	
	
	

	






	
	
	#np.save(os.path.join(video_vector_folder, vid_name+".npy"),flow)

			




