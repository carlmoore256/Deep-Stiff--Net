# Deep-Stiff--Net



Keep the dataset in the parent folder of the codes


1. extract_frames.py: This take all the videos, extract frames from it and stores it in a folder Stiffimages. The input would be .avi/.mov video files. The ouput will be individual colored images

2. pre_process.py: This performs preprocessing like cropping, cutting, and certain manipulations on the extracted frames to make it more relevant to our task. Cropping is done to make sure that the image size is reduced to be compatible with pre-trained networks like ResNet.


3. create_dataset.py: This creates the numpy vectors and the dataset for input to cnn.  It puts all the images in a matrix and stores the matrix in a .npy file.

4. cnn-resnet.py: Extract vectors from last layer of resnet and stores it as input. The flatten_1 layer of resnet is used to extract 2048 dimensional vector for each image and is used for further input.

5. cnn_final.py: this builds the own further model with two different losses and two outputs: One for stiffness, other for texture. The loss for regression is mean squared logarithmic error and classification has cross entropy loss .

6. Create_video_vec.py takes as input cut videos and returns numpy vectors of each video.
You can change the path to OpticalFlow to get the same for optical videos.

7. cnn+lstm.py: It trains the cnn+lstm network for double loss. Tuning pramameters can also be done using this coce.

8. cnn+lstm_bs.py: It trains the network for classification on stiffness. Stiffness is divided in 6 different classes . Input is a video and output is the resulting class for that video. Training and hyperparameter tuning can be done using this code for this model.

9. cnn+lstm_bs_regression.py: Trains the network for regression on stiffness. Input is a video with vectors extracted from resnet and output is the regression value of the stiffness.

10. create_triplets.py: Creates triplets where in all the three parameters : Mass, stiffness and texture are same for the positive and anchor

11. triplet_cnn+lstm.py: Trains using triplet loss. The rest of the model is a similar CNN+LSTM structure for videos. But,the final output is a 128 dimensional embedding for each video. We can use this to test new videos using triplet_test.py file as well.

12. sample.py: Samples triplets based on the criteria we desire. For our case, positive and anchor have all the three pramaters same. Negative has to have at least one parameter that is different.
