import cv2
import sys
import numpy as np 
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import LSTM, Dense
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate , concatenate, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
from keras.models import model_from_json

from keras import backend as K

from sample import sampling


timesteps=150
data_dim =2048
embed_size= 128
num_epochs=80


## Triplet loss for cnn+lstm
# path= "Videohogvectors"
# files= os.listdir(path)
# a=[]
# b=[]
# c=[]


# diction= {"Bro":0, "Cotton":1,"Felt": 2, "Sha":3,"Silk":4, "Sweater":5}


# for i, folders in enumerate(files):
# 	video_folder_path= os.path.join(path,folders)
# 	video_files= os.listdir(video_folder_path)
# 	for j, videos in enumerate(video_files):
# 		file_path=os.path.join(video_folder_path,videos)
# 		feature= np.load(file_path)
# 		print(folders,videos)

# 		vid_name=os.path.splitext(videos)[0]
# 		spt=vid_name.split("_")
# 		spt[1]=spt[1][1:]
# 		spt[2]=spt[2][1:]

# 		a.append(feature)
# 		b.append(float(spt[1]))
# 		c.append(diction[spt[0]])
# 		print(float(spt[1]),diction[spt[0]],spt[0])


# input_to_lstm=np.asarray(a)
# bs_label=np.asarray(b)
# texture_label=  np.asarray(c)
# texture_label=keras.utils.to_categorical(texture_label,6)


# bs_label=bs_label.reshape(len(bs_label),1)

# ## Extra dimag
# bs_label=np.log10(bs_label)
# bs_label=bs_label+2


# texture_label=np.append(texture_label,bs_label,axis=1)
# print(input_to_lstm.shape,bs_label.shape,texture_label.shape) 



# X_train, X_test, y_train, y_test = train_test_split( input_to_lstm,texture_label, test_size=0.2, shuffle= True, random_state=42)

# print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)

# input_train=X_train
# input_test=X_test
# bs_label_train= y_train[:,-1]
# bs_label_test=y_test[:,-1]
# texture_label_train=y_train[:,0:y_train.shape[1]-1]
# texture_label_test=y_test[:,0:y_test.shape[1]-1]

# print(input_train.shape,input_test.shape, bs_label_train.shape, bs_label_test.shape, texture_label_train.shape,texture_label_test.shape)

# #print(bs_label_train)

# timesteps=input_train.shape[1]
# data_dim= input_train.shape[2]
# num_classes=6


## Defining Triplet Loss
def triplet_loss(y_true, y_pred, alpha= 0.2):
## eEmbedding size
	embed_size= 128

	anchor_embed = y_pred[:,0:embed_size]
	positive_embed = y_pred[:, embed_size:2*embed_size]
	negative_embed = y_pred[:, 2*embed_size: 3*embed_size]

	pos_dist = K.sum(K.square(anchor_embed- positive_embed),axis=1)
	neg_dist = K.sum(K.square(anchor_embed- negative_embed),axis=1)

	diff_loss = pos_dist-neg_dist+alpha

	loss = K.mean(K.maximum(diff_loss,0.0),axis=0)

	return loss
 






## Create the triplet model
def create_model():

	anchor_input = Input(shape=(timesteps,data_dim), name='anchor_input')
	positive_input= Input(shape=(timesteps,data_dim), name= 'positive_input')
	negative_input= Input(shape=(timesteps,data_dim),name='negative_input')

	model_shared= Sequential()
	model_shared.add(LSTM(128, input_shape= (timesteps,data_dim), activation='relu',dropout=0.2, return_sequences= True))
	model_shared.add(LSTM(128, activation='relu',dropout=0.2))
	model_shared.add(Dense(128,activation= 'sigmoid'))

	print(model_shared.summary)
	# shared_LSTM_1 = LSTM(128, activation='relu',input_shape= (timesteps,data_dim),dropout=0.2, return_sequences= True)
	# shared_LSTM_2  = LSTM(128, activation='relu',dropout=0.2)(shared_LSTM_1)
	# shared_dense_1 = Dense(128,activation= 'sigmoid')(shared_LSTM_2)


	# anchor_1  = shared_dense_1(anchor_input)
	# positive_1 = shared_dense_1(positive_input)
	# negative_1 = shared_dense_1(negative_input)


	## Similar to siamese network
	anchor_1  = model_shared(anchor_input)
	positive_1 = model_shared(positive_input)
	negative_1 = model_shared(negative_input)

	## merge the 3 vectors
	merged_vector = concatenate([anchor_1, positive_1, negative_1],axis=-1)



	model = Model(inputs=[anchor_input, positive_input, negative_input], outputs= merged_vector)
	#adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer='adam',loss= triplet_loss)

	#callback=[keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0, mode='auto')]
	print(model.summary())

	return model
#print(model.summary())

## Store and save the model
model=create_model()
model_json = model.to_json()
with open("models/triplet/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5


## Define how many epochs you want to run for the triplet loss training
for epoch in range(num_epochs):

    print('Epoch %s' % epoch)

    # Sample triplets from the training data
    anchor, positive, negative = sampling()
    print(anchor.shape, positive.shape,negative.shape)
    X = {
        'anchor_input': anchor,
        'positive_input': positive,
        'negative_input': negative
    }
    callback=[keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0, mode='auto')]
    history=model.fit(X,
              np.zeros((anchor.shape[0],3*embed_size)),
              batch_size=64,
              epochs=1,
              shuffle=True,validation_split=0.2,callbacks=callback)


    ## save the model . second line for saving he model
    print(history.history['val_loss'])
    model.save_weights("models/triplet_optical/model_%d_val_loss_%f.h5"%(epoch,history.history['val_loss'][0]))
    print("Saved model to disk")




# history=model.fit({'anchor_input':anchor, 'positive_input': positive,'negative_input': negative},np.zeros((batch_size,3*embed_size))
# 	validation_data=({'main_input':input_test}, {'output_1': bs_label_test, 'output_2':texture_label_test}),
# 	epochs=100, batch_size=10)


# print(history.history['loss'])

# print(model.predict(input_test))
# print(bs_label_test)
# print(texture_label_test)