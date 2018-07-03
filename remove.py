import os
import shutil
import cv2
import numpy as np
# shutil.rmtree("StiffImages")

# os.rename("Optical_Path/b1/Sweater_wind2_b1_m0.3__1.avi","Optical_Path/b1/Sweater_wind2_b1_m0.3_1.avi")
# os.rename("Optical_Path/b1/Sweater_wind2_b1_m0.3__2.avi","Optical_Path/b1/Sweater_wind2_b1_m0.3_2.avi")
#Final_Dataset_6sec/b1/Sweater_wind2_b1_m0.3__2.avi 
# a=np.load('Videoopticalvectors/b0.001/Bro_wind3_b0.001_m1.3_2.npy')
# np.save('Videoopticalvectors/b0.001/Bro_wind3_b0.001_m1.3_1.npy',a)
#os.remove("Videoopticalvectors/b0.001/Bro_wind3_b0.001_m1.3_1.npy")
#shutil.rmtree("../Final_Dataset_6sec/b10Silk_wind2_b10_m0.7_1.avi")
# shutil.rmtree("Videovectors")

# print('I am here')

# path="../Texture_Motion_Preliminary"
# files= os.listdir(path)
# print(len(files))
# # for i in files:
# # 	folder_path=os.path.join(path,i)
# # 	files_inside=os.listdir(folder_path)
# # 	for j in files_inside:
# # 		if(j[0]=='.'):
# # 			os.remove(os.path.join(folder_path,j))


# for i in files:
# 	folder_path=os.path.join(path,i)
# 	files_inside=os.listdir(folder_path)
# 	print(len(files_inside))

## For printing the examples that are correct and that are wrong
def find(file_name):

	vid_name=os.path.splitext(file_name)[0]
	spt=vid_name.split("_")
	texture=spt[0]
	wind=spt[1]
	bs=spt[2]
	mass=spt[3]
	return texture, bs, mass, wind


anchor= np.load('bs_anchor.npy')
positive= np.load('bs_positive.npy')
negative=np.load('bs_negative.npy')

dist=np.load("final_dist.npy")
cnt=0
count=0
for i in range(0,len(dist)):
	texture_a, bs_a, mass_a, wind_a = find(anchor[i])
	texture_p, bs_p, mass_p, wind_p = find(positive[i])
	texture_n, bs_n, mass_n, wind_n = find(negative[i])
	print("yolo")
	if(texture_a==texture_n and bs_a==bs_n and mass_a==mass_n):
		count+=1
		#print(texture_a, texture_n, bs_a, bs_n, mass_a, mass_n, dist[i], wind_a, wind_n)
		if(dist[i]>0):
			print(texture_a, texture_n, bs_a, bs_n, mass_a, mass_n, dist[i], wind_a, wind_p, wind_n)
			cnt+=1
			#print
	# if(texture_a==texture_n and dist[i]>0):
	# 	print("kjdd")




print(count, cnt)

