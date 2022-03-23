The goal is to implement instance based learning on NSLKDD Dataset

#Preprocessing
##Run preprocess.py first
preprocess.py handles the preprocessing of the raw csv files into numpy arrays used for training and testing. First, we read in the csv to pandas dataframes and then to numpy arrays. Then, we get rid of the nominal attributes, namely elements 1,2,3. Then we split into x and y sets. The x sets have all of the binary and numerical attributes of the NSLKDD Datasets. The y sets have the class labels. 

For preparing our misuse detection sets, we encode the class labels as follows: normal = 0, DOS = 1, Probe = 2, R2L = 3, U2R = 4 and we leave the training sets as is, with a mix of normal and attack samples. For preparing our anomaly detection sets, we encode normal as 0 and attack as 1. We only include normal samples in the training sets.

The .npy files above represent our training and testing x and y sets. We do not include an anomaly_y_train, because all of our anomaly_x_train samples are normal, meaning our y labels would all be zero.
