import json
import os
import sys




## creates tripletsand saves to a json file.

## creates based on all 3 parameters to be same. Change if you want only stiffness
lst=[]
#path="vidvec"

lst=[]
# videos=os.listdir(path)
# print(videos[16],videos[42])
filename="triplet_file.json"

bs=[0.001, 0.01, 0.1, 1, 10, 100]
mass=[0.1, 0.3, 0.5, 0.7, 1, 1.3, 1.7]
texture=["Silk","Cotton","Fleece","FeltGreen","Sha","Sweater","Bro"]
for i in texture:
	for j in bs:
		for k in mass:
			diction={}
			diction[1]= i+"_"+"wind1"+"_"+"b"+str(j)+"_"+"m"+str(k)+"_"+str(1)+".npy"
			diction[2]= i+"_"+"wind1"+"_"+"b"+str(j)+"_"+"m"+str(k)+"_"+str(2)+".npy"
			diction[3]= i+"_"+"wind2"+"_"+"b"+str(j)+"_"+"m"+str(k)+"_"+str(1)+".npy"
			diction[4]= i+"_"+"wind2"+"_"+"b"+str(j)+"_"+"m"+str(k)+"_"+str(2)+".npy"
			diction[5]= i+"_"+"wind3"+"_"+"b"+str(j)+"_"+"m"+str(k)+"_"+str(1)+".npy"
			diction[6]= i+"_"+"wind3"+"_"+"b"+str(j)+"_"+"m"+str(k)+"_"+str(2)+".npy"
			lst.append(diction)




with open(filename,'w+') as fp:
	json.dump(lst,fp)
