import cv2
import sys
import numpy as np 
#from matplotlib import pyplot as plt 
import os

import time
start_time = time.time()

## Cuts the video in half
width=1280
height=720


path="../final_dataset"

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
				else :
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

			




