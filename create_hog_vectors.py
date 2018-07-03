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
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model


## Calculates hog features from images

## Input an images, get its hog feature vector which is 576 dimensional

## if input a video, get a sequence of hog vectors. 
def find_hog(image):
    size=image.shape
    #print(size)
    winSize = (image.shape[0],image.shape[1])
    blockSize = (20,20)
    blockStride = (10,10)
    cellSize = (20,20)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True 
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,
        L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    descriptor = hog.compute(image)
    #print(descriptor.shape)
    return descriptor





def extract_frame():
	frate=24
	delay= int(1000/24)
	path= "../Texture_Motion_Preliminary"
	files=os.listdir(path)
	#files=[ 'b0.05','b50','b100','b180','b0.0005','b40','b0.005','b450','b110','b0.5','b60','b0.1','b25','b1', 'b600']
	#files=os.listdir(path)
	#print(files)
	video_file_path= "Videohogvectors"
	# model = ResNet50(weights='imagenet')
	# model_new = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
	for i,folders in enumerate(files):
		folder_path=os.path.join(path,folders)
		class_files= os.listdir(folder_path)
		video_vector_folder= os.path.join(video_file_path,folders)
		if not os.path.exists(video_vector_folder):

			os.makedirs(video_vector_folder)


		cnt=0
		for j, videos in enumerate(class_files):
			video_path= os.path.join(folder_path,videos)
			vid_file=cv2.VideoCapture(video_path)
			lst=[]
			vid_name=os.path.splitext(videos)[0]
			while(vid_file.isOpened()):
				ret, frame = vid_file.read()
				if (ret==True):
					img=frame[100:650,300:950]
					img = cv2.resize(img,(150,150))
					feature= find_hog(img)
					lst.append(feature)
					if (cv2.waitKey(25) & 0xFF == ord('q')):
						break
				else:
					break


			lst=np.asarray(lst)
			lst=np.squeeze(lst)
			print(lst.shape,video_path)
			#print(folders,videos,lst.shape)
			np.save(os.path.join(video_vector_folder, vid_name+".npy"),lst)
			cnt+=1





	vid_file.release()

	cv2.destroyAllWindows()




if not os.path.exists("Videohogvectors"):
	os.makedirs("Videohogvectors")


extract_frame()

# input_data= np.load('inputs.npy')
# input_data = input_data.reshape(input_data.shape[0],250,250,3)
# input_data=input_data[:,13:237,13:237,:]
# print(input_data.shape)
# labels= np.load('label.npy')
# labels=keras.utils.to_categorical(labels,21)
# print(labels.shape)

# X_train, X_test, y_train, y_test = train_test_split( input_data, labels, test_size=0.2, shuffle= True, random_state=42)

# print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)
# X_train_batch_1= X_train[0:100]
# y_train_batch=y_train[0:100]
# # print(X_train_batch.shape, y_train_batch.shape, X_test.shape,y_test.shape)
# # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(250,250,3))
# # result= base_model.predict_on_batch(X_train_batch)
# # print(result.shape)
# # print(base_model.summary())

# model = ResNet50(weights='imagenet')

# # img_path = 'elephant.jpg'
# # img = image.load_img(img_path, target_size=(224, 224))
# # x = image.img_to_array(img)
# # print(x.shape)

# # x = np.expand_dims(x, axis=0)
# # print(x.shape)
# #x = preprocess_input(x)

# #preds = model.predict(x)
# model_new = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
# pred=model_new.predict(X_train_batch)

# #print(preds)
# #print(model.summary(), model.layers[28].output)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print(pred.shape)

# model = Sequential()
# model.add(LSTM(32, return_sequences=True,
#                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(32))  # return a single vector of dimension 32
# model.add(Dense(10, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# # Generate dummy training data
# x_train = np.random.random((1000, timesteps, data_dim))
# y_train = np.random.random((1000, num_classes))
# #print(x_train)

# # Generate dummy validation data
# x_val = np.random.random((100, timesteps, data_dim))
# y_val = np.random.random((100, num_classes))

# model.fit(x_train, y_train,
#           batch_size=64, epochs=5,
#           validation_data=(x_val, y_val))
