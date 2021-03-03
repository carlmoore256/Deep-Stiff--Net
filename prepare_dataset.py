# extract frames from videos, crop and pre-process, save as dataset npy
import os
import glob
import cv2
import numpy as np

# custom cropping for the cloth dataset
def crop_frame(frame, bounds=[100,650,300,950]):
    return frame[bounds[0]:bounds[1], bounds[2]:bounds[3]]

def resize_frame(frame, new_size=(250,250)):
    return cv2.resize(frame, new_size)

def pre_process(frame):
    frame = crop_frame(frame)
    frame = resize_frame(frame)
    return frame

# extracts frames from an individual file, set minMaxFrames as
# upper and lower bounds of frames to extract from video
def extract_frames(file, minMaxFrames=[10,140]):
    frames = []
    video = cv2.VideoCapture(file)
    idx = 0

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if idx >= minMaxFrames[0] and idx < minMaxFrames[1]:
                frame = pre_process(frame)
                frames.append(frame)
            idx += 1
        else:
            break
    return frames


# give a filepath, return cloth dataset labels "bs_label" and "texture_label"
def labels_from_path(path, label_dict):
    filename = os.path.basename(path)
    split = filename.split("_")
    bs_label = float(split[2][1:]) # remove first char
    texture_label = label_dict[split[0]]
    return bs_label, texture_label


# creates a npy file of image and label pairs from a single path
def dataset_from_videos(path, label_dict, output_path="./dataset", vid_ext = ".avi"):
    # if not os.path.exists("./tmp"):
    #     os.path.create("./tmp")

    for directory in os.listdir(path):
        video_files = glob.glob(f"{os.path.join(path, directory)}/*{vid_ext}")
        frames = []
        bs_labels = []
        texture_labels = []

        for f in video_files:
            print(f'extracting frames and labels from {f}')
            bs_label, texture_label = labels_from_path(f, label_dict)
            print(f'LABELS: bs_label {bs_label}, texture {texture_label}')
            bs_labels.append(bs_label)
            texture_labels.append(texture_label)
            frames += extract_frames(f)

    

    frames = np.asarray(frames)
    print(frames.shape)

    np.save(f"{output_path}/images.npy", frames)
    np.save(f"{output_path}/bs_labels.npy", bs_labels)
    np.save(f"{output_path}/texture_labels.npy", texture_labels)

if __name__ == '__main__':

    label_dict = {  "Bro":0, 
                    "Cotton":1,
                    "FeltGreen": 2, 
                    "Sha":3,
                    "Silk":4, 
                    "Sweater":5,
                    "Fleece":6
                 }

    dataset_path = "/Volumes/RICO_III/ClothDataset/Final_Dataset_6sec"
    dataset_from_videos(dataset_path, 
                        label_dict,
                        output_path="/Volumes/RICO_III/ClothDataset")
