# extract features of images from resnet, save a npy

import numpy as np 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

dataset_path = "E:\Datasets\ClothStiffnessDataset"

## Finds features using resnet
input_data= np.load(f'{dataset_path}/images.npy')

# clip data for resnet size
input_data=input_data[:,13:237, 13:237,:]
print(input_data.shape)

## load ResNet

## Extract features from 2nd last layer of ResNet
model = ResNet50(weights='imagenet')
model_new = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
print('Prediction Starts')
feature_extracted=model_new.predict(input_data,verbose=2)
np.save('feature.npy',feature_extracted)
print("prediction ends")