import cv2
import sys
import numpy as np 
#from matplotlib import pyplot as plt 
import os
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.applications.resnet50 import preprocess_input, decode_predictions





## Input is a numpy array of dimensions (No. of images, imagedimensions)

## Finds features using resnet
input_data= np.load('inputs.npy')
input_data = input_data.reshape(input_data.shape[0],250,250,3)
input_data=input_data[:,13:237,13:237,:]
print(input_data.shape)
#input_data = preprocess_input(input_data)


bs_label= np.load('bs_label.npy')
texture_label=np.load('texture_label.npy')


texture_label=keras.utils.to_categorical(texture_label,7)
print(texture_label.shape)


## load ResNet

## Extract features from 2nd last layer of ResNet
model = ResNet50(weights='imagenet')
model_new = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
print('Prediction Starts')
feature_extracted=model_new.predict(input_data,verbose=2)
np.save('feature.npy',feature_extracted)
print("prediction ends")
# feature_extracted=np.load('feature.npy')
# labels= np.load('label.npy')
# labels=keras.utils.to_categorical(labels,21)
# X_train, X_test, y_train, y_test = train_test_split( feature_extracted, labels, test_size=0.2, shuffle= True, random_state=42)

# print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)

# def cnn_model():

# 	num_classes=21
# 	model=Sequential()
# 	model.add(Dense(1024, activation='relu',input_dim= 2048))
# 	model.add(Dense(512, activation='relu'))
# 	model.add(Dense(128, activation='relu'))
# 	model.add(Dense(32, activation='relu'))
# 	model.add(Dense(num_classes, activation='softmax'))
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 	return model

# model_forward=cnn_model()
# #plot_model(model, to_file='model.png')

# model_forward.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=20)
# scores = model_forward.evaluate(X_test, y_test, verbose=0)
# print(scores)
# model_json = model.to_json()
# with open("./model_4000_binary1.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_4000_binary1.h5")
# print("Saved model to disk")
