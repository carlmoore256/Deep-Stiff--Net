


## Code for training the CNN with images

## Input is a feature vector for each image that is extracted from ResNet.

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
from keras.layers import Input, Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model





# Number of classes of textures
num_classes=7

# Load feature vectors of images and bending stiffness values, texture values
feature_extracted=np.load('feature_mean.npy')
bs_label= np.load('bs_label.npy')
texture_label= np.load('texture_label.npy')

# Convert texture to onehot vector (make it 7 dimensional vector)
texture_label=keras.utils.to_categorical(texture_label,7)

bs_label=bs_label.reshape(len(bs_label),1)
feature_extracted=np.append(feature_extracted,bs_label,axis=1)
print(feature_extracted.shape,bs_label.shape,texture_label.shape)



## Separate the test set
X_train, X_test, y_train, y_test = train_test_split( feature_extracted,texture_label, test_size=0.1, shuffle= True, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)


## Let the code stay as it is. Dont mess it up. You will finally get the traning dataset for sure.
input_train=X_train[:,0:X_train.shape[1]-1]
input_test=X_test[:,0:X_test.shape[1]-1]
bs_label_train= X_train[:,-1]
bs_label_test=X_test[:,-1]
texture_label_train=y_train
texture_label_test=y_test



print(input_train.shape,input_test.shape, bs_label_train.shape, bs_label_test.shape, texture_label_train.shape,texture_label_test.shape)


## Starts the CNN Model with output1 and output2 being 2 outputs

main_input = Input(shape=(2048,), name='main_input')
x = Dense(1024, activation='relu')(main_input)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output_1= Dense(1,name='output_1')(x)
output_2= Dense(num_classes,name='output_2',activation='softmax')(x)

## Compile the model, feel free to use different loss_weights for tuning
model = Model(inputs=[main_input], outputs=[output_1, output_2])
model.compile(optimizer='adam',
              loss={'output_1': 'mean_squared_logarithmic_error', 'output_2': 'categorical_crossentropy'},
              loss_weights={'output_1': 0.5, 'output_2': 0.5})
#plot_model(model, to_file='model_cnn.png')

##  Commentout  call backs if you dont want your model to stop early. Also comment out call backs from model.fit

## Check more about callbacks on internet. 
callback=[keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0, mode='auto')]

#print(model.summary())
# history=model.fit({'main_input':input_train},
#           {'output_1': bs_label_train, 'output_2':texture_label_train},
#           validation_data=({'main_input':input_test}, {'output_1': bs_label_test, 'output_2':texture_label_test}),
#           epochs=100, batch_size=64,callbacks=callback)

## Train the model
history=model.fit({'main_input':input_train},
          {'output_1': bs_label_train, 'output_2':texture_label_train},
          validation_split=0.2, shuffle = True,
          epochs=100, batch_size=64,callbacks=callback)

## histpry stores training and validation loss in a numpy array

## Just things to test your results. Can comment out while training. 
## Devise your own codes here for your own comfort
print(history.history['loss'])
results=model.predict(input_test)
texture_results= results[1]
bs_results=results[1]
match= np.sum(np.argmax(texture_results,axis=1)==np.argmax(texture_label_test))
print(match)

bs_results=results[0]
bs_results=np.squeeze(bs_results)
print(bs_results- bs_label_test )

#print(model.predict(input_test))

#print(bs_label_test,texture_label_test)

# scores = model_forward.evaluate(X_test, y_test, verbose=0)
# print(scores)
# model_json = model.to_json()
# with open("./model_4000_binary1.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_4000_binary1.h5")
# print("Saved model to disk")
