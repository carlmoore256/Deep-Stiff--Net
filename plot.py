import cv2
import sys
import numpy as np 
import matplotlib 
#matplotlib.use('agg')
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


## Can be helpfu in plotting sometimes

# num_classes=7
# main_input = Input(shape=(2048,), name='main_input')
# x = Dense(1024, activation='relu')(main_input)
# x = Dense(512, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# output_1= Dense(1,name='output_1')(x)
# output_2= Dense(num_classes,name='output_2',activation='softmax')(x)


# model = Model(inputs=[main_input], outputs=[output_1, output_2])
# model.compile(optimizer='adam',
#               loss={'output_1': 'mean_squared_logarithmic_error', 'output_2': 'categorical_crossentropy'},
#               loss_weights={'output_1': 0.5, 'output_2': 0.5})
def triplet_loss(y_true, y_pred, alpha= 0.2):

	#embed_size= 100

	anchor_embed = y_pred[:,0:embed_size]
	positive_embed = y_pred[:, embed_size:2*embed_size]
	negative_embed = y_pred[:, 2*embed_size: 3*embed_size]

	pos_dist = K.sum(K.square(anchor_embed- positive_embed),axis=1)
	neg_dist = K.sum(K.square(anchor_embed- negative_embed),axis=1)

	diff_loss = pos_dist-neg_dist+alpha

	loss = K.mean(K.maximum(diff_loss,0.0),axis=0)

	return loss
 

timesteps=150
data_dim =2048
embed_size= 128
num_epochs=80

num_classes=7

main_input = Input(shape=(timesteps,data_dim), name='main_input')

#### Earlier Model
# x = LSTM(96, activation='relu')(main_input)
# x = Dense(32, activation='sigmoid')(x)
# output_1= Dense(1,name='output_1')(x)
# output_2= Dense(num_classes,name='output_2',activation='softmax')(x)

# x =LSTM(128, activation='relu',dropout=0.2, return_sequences= True)(main_input)
# x = LSTM(128, activation='relu',dropout=0.2)(x)
# x = Dense(64,activation= 'relu')(x)
# x = Dense(32,activation= 'relu')(x)
# output_1= Dense(1,name='output_1')(x)
# output_2= Dense(num_classes,name='output_2',activation='softmax')(x)




# model = Model(inputs=[main_input], outputs=[output_1, output_2])
# adam=keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# #sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=adam,
#               loss={'output_1': 'mean_squared_logarithmic_error', 'output_2': 'categorical_crossentropy'},
#               loss_weights={'output_1': 0.5, 'output_2': 0.5})




# print(model.summary())

# anchor_input = Input(shape=(timesteps,data_dim), name='anchor_input')
# positive_input= Input(shape=(timesteps,data_dim), name= 'positive_input')
# negative_input= Input(shape=(timesteps,data_dim),name='negative_input')

# model_shared= Sequential()
# model_shared.add(LSTM(128, input_shape= (timesteps,data_dim), activation='relu',dropout=0.2, return_sequences= True))
# model_shared.add(LSTM(128, activation='relu',dropout=0.2))
# model_shared.add(Dense(128,activation= 'sigmoid'))

# print(model_shared.summary)
# # shared_LSTM_1 = LSTM(128, activation='relu',input_shape= (timesteps,data_dim),dropout=0.2, return_sequences= True)
# # shared_LSTM_2  = LSTM(128, activation='relu',dropout=0.2)(shared_LSTM_1)
# # shared_dense_1 = Dense(128,activation= 'sigmoid')(shared_LSTM_2)


# # anchor_1  = shared_dense_1(anchor_input)
# # positive_1 = shared_dense_1(positive_input)
# # negative_1 = shared_dense_1(negative_input)

# anchor_1  = model_shared(anchor_input)
# positive_1 = model_shared(positive_input)
# negative_1 = model_shared(negative_input)


# merged_vector = concatenate([anchor_1, positive_1, negative_1],axis=-1)



# model = Model(inputs=[anchor_input, positive_input, negative_input], outputs= merged_vector)
# adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(optimizer=adam,loss= triplet_loss)

# #callback=[keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0, mode='auto')]
# print(model.summary())
# plot_model(model, to_file='img_pres/triplet_cnn+lstm_1.png', show_shapes= False)

# val_loss=[1.6248,1.0312,0.7329,0.6412, 0.3467, 0.2996, 0.2904, 0.1648, 0.2139, 0.1162, 0.0911, 0.0779, 0.1005, 0.0461, 
# 0.0681, 0.0342, 0.0920, 0.0164, 0.0293, 0.0131, 0.0130, 0.0335, 0.0297, 0.0235, 0.0096, 0.0107, 0.0135, 0.0106, 0.0135,
# 0.0231, 0.0306, 0.0186,0.0073, 0.0019, 0.0060, 0.0017,0.118, 0.0013, 0.0014, 0.000795 ]

# train_loss= [2.2159, 1.4037, 0.9988, 0.5865, 0.5428, 0.3669, 0.2927, 0.2280, 0.1578, 0.1786, 0.1349, 0.1276, 0.1080,
# 0.0932, 0.0719, 0.0682, 0.0534, 0.0666, 0.0429, 0.0504, 0.0389, 0.0476, 0.0423, 0.0216, 0.0210, 0.0283, 0.0165,
# 0.0178, 0.136, 0.0173, 0.0144, 0.0194, 0.0091, 0.0123, 0.0138, 0.0099, 0.0100, 0.0132, 0.0127, 0.0122]
# print(len(train_loss))
# epoch = list(range(1,51))
# print(epoch)

# loss= np.loadtxt("loss_20_50_Adam.txt")
# loss=loss.tolist()
# val_loss= np.loadtxt("val_loss_20_50_Adam.txt")
# val_loss= val_loss.tolist()
# #print(loss.shape)

# plt.plot(epoch, val_loss,'r', label="Validation Loss")
# plt.plot(epoch, loss,'g',label="Training Loss")
# plt.legend(loc='upper right')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

