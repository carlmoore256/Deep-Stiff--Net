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



## Just some random temporary file
anchor_input = Input(shape=(timesteps,data_dim), name='anchor_input')
positive_input= Input(shape=(timesteps,data_dim), name= 'positive_input')
negative_input= Input(shape=(timesteps,data_dim),name='negative_input')


def generate_data(directory, batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    file_list = os.listdir(directory)
    while True:
        image_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            sample = file_list[i]
            i += 1
            image = cv2.resize(cv2.imread(sample[0]), INPUT_SHAPE)
            image_batch.append((image.astype(float) - 128) / 128)

        yield np.array(image_batch)