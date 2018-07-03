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
from keras.optimizers import SGD

## performs just regression on bending stiffness for videos
path= "Videovectors"
files= os.listdir(path)
a=[]
b=[]
c=[]


#diction= {"Bro":0, "Cotton":1,"FeltGreen": 2, "Sha":3,"Silk":4, "Sweater":5, "Fleece":6}
diction= {"0.001":0, "0.01":1,"0.1": 2, "1":3,"10":4, "100":5}

## Load the videos
for i, folders in enumerate(files):
	video_folder_path= os.path.join(path,folders)
	video_files= os.listdir(video_folder_path)
	for j, videos in enumerate(video_files):
		file_path=os.path.join(video_folder_path,videos)
		feature= np.load(file_path)
		#print(folders,videos)
		#feature_shape=feature.shape
		vid_name=os.path.splitext(videos)[0]
		spt=vid_name.split("_")
		spt[2]=spt[2][1:]
		spt[3]=spt[3][1:]
		#print(feature.shape,folders,videos)
		#feature=np.asarray(feature)
		#print(type(feature))
		a.append(feature)
		b.append(diction[spt[2]])
		#print(diction[spt[2]],spt[2])

#print(np.asarray(a[10]).shape)
print(type(a))
input_to_lstm = np.asarray(a)
#input_to_lstm = only_for_optical(input_to_lstm)
print(input_to_lstm.shape)
bs_label=np.asarray(b)
bs_label=keras.utils.to_categorical(bs_label,6)


## Extra dimag
#bs_label=np.log10(bs_label)
#bs_label=bs_label+3
#bs_label =(bs_label-np.min(bs_label))/(np.max(bs_label)-np.min(bs_label))
#print(input_to_lstm.shape,bs_label.shape) 


print(input_to_lstm.shape)
X_train, X_test, y_train, y_test = train_test_split( input_to_lstm,bs_label, test_size=0.1, shuffle= True, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)

input_train=X_train
input_test=X_test
bs_label_train= y_train
bs_label_test=y_test

print(input_train.shape,input_test.shape, bs_label_train.shape, bs_label_test.shape)
## Create the model
#print(bs_label_train)
def create_model(algo):
	timesteps=input_train.shape[1]
	data_dim= input_train.shape[2]
	print(timesteps,data_dim)
	num_classes=6

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
	#output_1= Dense(1,name='output_1')(x)
	output_2= Dense(num_classes,name='output_2',activation='softmax')(x)




	model = Model(inputs=main_input, outputs=output_2)
	adam=keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	sgd = SGD(lr=0.1, decay = 1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',optimizer=algo,metrics=['accuracy'])


	print(model.summary())
	return model



#model=create_model()
#callback=[keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0, mode='auto')]
#history=model.fit(input_train,bs_label_train, batch_size=32, epochs=50,validation_split=0.2,shuffle=True)


## these are 3 hyperparameters list used for tuning. Feel free to use them accordingly
## keep only one element per list for elementary training
batch_size = [10,20,40,80,100]
epochs = [20,50,100]
optimizer=['SGD', 'RMSprop', 'Adagrad','Adam']
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

			## Comment out below 2 lines if you need to save the loss
			# np.savetxt("models/cnn+lstm_bs/loss_%d_%d_%s.txt" %(batch,epoch_num,algo),normal_loss, delimiter=",")
			# np.savetxt("models/cnn+lstm_bs/val_loss_%d_%d_%s.txt" %(batch,epoch_num,algo), v_loss, delimiter=",")

# v_loss=history.history['val_loss']
# v_loss=np.array(v_loss)
# normal_loss=history.history['loss']
# normal_loss=np.array(normal_loss)
# np.savetxt("models/cnn+lstm/loss_%d_%d_%s.txt" %(batch,epoch_num,algo),normal_loss, delimiter=",")
# np.savetxt("models/cnn+lstm/val_loss_%d_%d_%s.txt" %(batch,epoch_num,algo), v_loss, delimiter=",")


# # model = Model(inputs=main_input, outputs=output_2)
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# history=model.fit(input_train, texture_label_train, batch_size=32, epochs=10,validation_split=0.2,shuffle=True)



# print(history.history['loss'])

# print(model.evaluate(input_test))
# print(bs_label_test)
# print(texture_label_test)