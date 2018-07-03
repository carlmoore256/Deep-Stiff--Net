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
from keras.layers import Input, Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


## Input the video data from here.
path= "Videovectors"
files= os.listdir(path)
a=[]
b=[]
c=[]


diction= {"Bro":0, "Cotton":1,"FeltGreen": 2, "Sha":3,"Silk":4, "Sweater":5, "Fleece":6}

## load input
for i, folders in enumerate(files):
	video_folder_path= os.path.join(path,folders)
	video_files= os.listdir(video_folder_path)
	for j, videos in enumerate(video_files):
		file_path=os.path.join(video_folder_path,videos)
		#print(file_path)
		feature= np.load(file_path)
		#print(folders,videos)

		vid_name=os.path.splitext(videos)[0]
		spt=vid_name.split("_")
		spt[2]=spt[2][1:]
		spt[3]=spt[3][1:]

		a.append(feature)
		b.append(float(spt[2]))
		c.append(diction[spt[0]])
		#print(float(spt[2]),diction[spt[0]],spt[0])


input_to_lstm=np.asarray(a)
bs_label=np.asarray(b)
texture_label=  np.asarray(c)
texture_label=keras.utils.to_categorical(texture_label,7)


bs_label=bs_label.reshape(len(bs_label),1)

## Extra dimag
#bs_label=np.log10(bs_label)
#bs_label=bs_label+3
#bs_label =(bs_label-np.min(bs_label))/(np.max(bs_label)-np.min(bs_label))

texture_label=np.append(texture_label,bs_label,axis=1)
print(input_to_lstm.shape,bs_label.shape,texture_label.shape) 

## Split into test test

X_train, X_test, y_train, y_test = train_test_split( input_to_lstm,texture_label, test_size=0.1, shuffle= True, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)

input_train=X_train
input_test=X_test
bs_label_train= y_train[:,-1]
bs_label_test=y_test[:,-1]
texture_label_train=y_train[:,0:y_train.shape[1]-1]
texture_label_test=y_test[:,0:y_test.shape[1]-1]

print(input_train.shape,input_test.shape, bs_label_train.shape, bs_label_test.shape, texture_label_train.shape,texture_label_test.shape)


## Model defining
#print(bs_label_train)
def create_model(algo):
	timesteps=input_train.shape[1]
	data_dim= input_train.shape[2]
	num_classes=7

	main_input = Input(shape=(timesteps,data_dim), name='main_input')

	#### Earlier Model
	# x = LSTM(96, activation='relu')(main_input)
	# x = Dense(32, activation='sigmoid')(x)
	# output_1= Dense(1,name='output_1')(x)
	# output_2= Dense(num_classes,name='output_2',activation='softmax')(x)

	x =LSTM(128, activation='relu',dropout=0.2, return_sequences= True)(main_input)
	x = LSTM(128, activation='relu',dropout=0.2)(x)
	x = Dense(64,activation= 'relu')(x)
	x = Dense(32,activation= 'relu')(x)
	output_1= Dense(1,name='output_1')(x)
	#output_2= Dense(num_classes,name='output_2',activation='softmax')(x)




	model = Model(inputs=main_input, outputs=output_1)
	adam=keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_logarithmic_error',optimizer=algo)

	print(model.summary())
	return model
## Hyperparameter list. keep only value in list per parameter for elementary training
batch_size = [10,20,40,80,100]
epochs = [20,50,100]
optimizer=['SGD', 'RMSprop', 'Adam', 'Adagrad']
#optimizer=['Adam']
for batch in batch_size:
	for epoch_num in epochs:
		for algo in optimizer:
			model=create_model(algo)
			#model=create_model()
			history=model.fit(input_train, bs_label_train, batch_size=batch, epochs=epoch_num,validation_split=0.2,shuffle=True)
			v_loss=history.history['val_loss']
			v_loss=np.array(v_loss)
			normal_loss=history.history['loss']
			normal_loss=np.array(normal_loss)

			## Comment if want to save loss
			# np.savetxt("models/cnn+lstm_bs_regression/loss_%d_%d_%s.txt" %(batch,epoch_num,algo),normal_loss, delimiter=",")
			# np.savetxt("models/cnn+lstm_bs_regression/val_loss_%d_%d_%s.txt" %(batch,epoch_num,algo), v_loss, delimiter=",")




# model = Model(inputs=main_input, outputs=output_2)
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# history=model.fit(input_train, texture_label_train, batch_size=32, epochs=10,validation_split=0.2,shuffle=True)






# print(history.history['loss'])

# print(model.evaluate(input_test))
# print(bs_label_test)
# print(texture_label_test)