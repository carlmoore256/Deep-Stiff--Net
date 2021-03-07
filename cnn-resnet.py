# extract features of images from resnet, save a npy
import tensorflow as tf
import numpy as np 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import argparse
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default= "E:\Datasets\ClothStiffnessDataset", help="path to dataset")
parser.add_argument("--outdir", type=str, default="./resnet_features", help="output directory of features")
parser.add_argument("--mean", type=bool, default=True, help="compute output feature mean")
args = parser.parse_args()

# not sure which layer to take since layer names have since changed
# after the original repo (original called for "flatten_1")
resnet_feature_layer = "conv5_block2_out"

output_file = f"resnet_features_{resnet_feature_layer}.npy"

if args.mean:
    output_file = f"resnet_features_mean_{resnet_feature_layer}.npy"

## Finds features using resnet
input_data= np.load(f'{args.dataset}/images.npy')
print(f"loaded input data, shape {input_data.shape}")

# clip data for resnet size
print("cropping input data to fit resnet50")
input_data=input_data[:,13:237, 13:237,:]

if args.mean:
    print(f'preprocessed images for resnet50')
    input_data = preprocess_input(input_data)

## load ResNet and Extract features from 2nd to last layer of ResNet
model = ResNet50(weights='imagenet')
tf.keras.utils.plot_model(model, to_file="model.png")
model_new = Model(inputs=model.input, outputs=model.get_layer('conv5_block2_out').output)

# generate model prediction
print(f"running resnet prediction on {input_data.shape[0]} images")
feature_extracted=model_new.predict(input_data,verbose=2)
np.save(output_file,feature_extracted)
print("features saved")