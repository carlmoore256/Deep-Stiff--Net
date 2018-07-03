



Keep the dataset in the parent folder of the codes


1. extract_frames.py: This take all the videos, extrcat frames from it and stores it in a folder Stiffimages.

2. pre_process.py: This performs preprocessing like cropping, cutting, and certain manipulations on the extracted frames to make it more relevant to our task.


3. create_dataset.py: This creates the numpy vectors and the dataset for input to cnn

4. cnn-resnet.py: Extract vectors from last layer of resnet and stores is as input.

5. cnn_final.py: this builds the own further model with two different losses and two outputs: One for stiffness, other for texture.

6. Create_video_vec.py takes as input cut videos and returns numpy vectors of each video.
You can change the path to OpticalFlow to get the same for optical videos.

7. cnn+lstm.py: It trains the cnn+lstm network for double loss

8. cnn+lstm_bs.py: It trains the network for classification on stiffness

9. cnn+lstm_bs_regression.py: It trains the network for regression on stiffness

10. create_triplets.py: Creates triplets

11. triplet_cnn+lstm.py: Trains triplet

12. sample.py: Samples triplets
