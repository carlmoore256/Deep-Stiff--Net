
import json
import random
import numpy as np
import os



## Samples the images for the triplet loss

## print some variables to get hold of the code

## Change it accordingly to fit for triplet only with stiffness as a parameter
def find_folder(file_name):
	vid_name=os.path.splitext(file_name)[0]
	spt=vid_name.split("_")
	folder_name= spt[2]
	bs_val=float(spt[2][1:])
	bs_path=os.path.join("Videovectors",folder_name)
	final_name= os.path.join(bs_path,file_name)

	return final_name,bs_val


## sample anchor, positive, negative accordingly

def sampling():

	filename="triplet_file.json"
	big_folder_path="Videovectors"
	with open(filename,"r") as fp:
		data=json.load(fp)

	data_size=len(data)
	training_size=100
	anchor_input= []
	positive_input =[]
	negative_input=[]
	# bs_anchor=[]
	# bs_positive=[]
	# bs_negative=[]

	## Select anchor , positive, negative according to certain conditions
	for i in range(0,training_size):


		select_list= list(range(0,data_size))
		random.shuffle(select_list)
		major= select_list.pop()
		minor= select_list.pop()
		#print(major,minor)
		anchor= data[major][str(1)]
		anchor, bs_anch=find_folder(anchor)

		pos_number= random.randint(3,6)

		positive=data[major][str(pos_number)]
		positive, bs_pos=find_folder(positive)

		neg_number= random.randint(1,6)

		negative= data[minor][str(neg_number)]
		negative, bs_neg= find_folder(negative)

		print(anchor, positive, negative)
		#print(anchor,positive,negative)
		anchor_input.append(np.load(anchor))
		positive_input.append(np.load(positive))
		negative_input.append(np.load(negative))
		# bs_anchor.append(bs_anch)
		# bs_negative.append(bs_neg)
		# bs_positive.append(bs_pos)





	anchor_input=np.asarray(anchor_input)
	positive_input=np.asarray(positive_input)
	negative_input=np.asarray(negative_input)
	# np.save('bs_anchor.npy',np.asarray(bs_anchor))
	# np.save('bs_positive.npy',np.asarray(bs_positive))
	# np.save('bs_negative.npy',np.asarray(bs_negative))

	# positive_input=np.asarray(positive_input)
	# negative_input=np.asarray(negative_input)
	print(anchor_input.shape,positive_input.shape,negative_input.shape)
	return anchor_input, positive_input, negative_input


#sampling()


