import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default= "E:\Datasets\ClothStiffnessDataset", help="path to dataset")
parser.add_argument("--outdir", type=str, default="./resnet_features", help="output directory of features")
parser.add_argument("--mean", type=bool, default=True, help="compute output feature mean")
args = parser.parse_args()

input_data = np.load(f'{args.dataset}/images.npy')
