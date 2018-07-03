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

from sample_test import sampling


timesteps=150
data_dim =2048
embed_size= 128
num_epochs=40



def error_calc(results):

	anchor_embed = results[:,0:embed_size]
	positive_embed = results[:, embed_size:2*embed_size]
	negative_embed =results[:, 2*embed_size: 3*embed_size]
	np.save('anchor_array.npy',anchor_embed)
	np.save('positive_array.npy',positive_embed)
	np.save('negative_array.npy',negative_embed)
	


	# pos_dist = K.sum(K.square(anchor_embed- positive_embed),axis=1)
	# neg_dist = K.sum(K.square(anchor_embed- negative_embed),axis=1)
	pos_dist = np.sum(np.square(anchor_embed- positive_embed),axis=1)
	neg_dist = np.sum(np.square(anchor_embed- negative_embed),axis=1)
	dist=pos_dist-neg_dist
	print(np.sum(dist<=0))
	return dist
	


# def get_vectors():


json_file = open('models/triplet/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/triplet/model_15_val_loss_0.034208.h5")
print("Loaded model from disk")

# anchor= get_vectors()
# positive=get_vectors()
# negative=get_vectors()

anchor, positive, negative = sampling()

results=loaded_model.predict([anchor,positive,negative])

final_results= error_calc(results)
np.save('final_dist.npy', final_results)
print(final_results)