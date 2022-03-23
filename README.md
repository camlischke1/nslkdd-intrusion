The goal is to implement instance based learning on NSLKDD Dataset

# Preprocessing

## Run preprocess.py first

preprocess.py handles the preprocessing of the raw csv files into numpy arrays used for training and testing. First, we read in the csv to pandas dataframes and then to numpy arrays. Then, we get rid of the nominal attributes, namely elements 1,2,3. Then we split into x and y sets. The x sets have all of the binary and numerical attributes of the NSLKDD Datasets. The y sets have the class labels. 

For preparing our misuse detection sets, we encode the class labels as follows: normal = 0, DOS = 1, Probe = 2, R2L = 3, U2R = 4 and we leave the training sets as is, with a mix of normal and attack samples. For preparing our anomaly detection sets, we encode normal as 0 and attack as 1. We only include normal samples in the training sets.

The .npy files above represent our training and testing x and y sets. We do not include an anomaly_y_train, because all of our anomaly_x_train samples are normal, meaning our y labels would all be zero.


# Anomaly Detection
After running preprocess.py and extracting all normal instances for the training data set, we find the nearest neighbor for each instance in the testing dataset using sklearn's knearest function. We vary the control threshold 10 times, starting from a distance of 10 to 100 in intervals of ten distance points. Anything greater than the threshold will be considered an anomaly. 

We calculate the FPR and TPR for each control threshold, and then plot the full ROC curve and get the area under the ROC curve. 

```bash
threshold = 10, FPR: 0.341, TPR: 0.511, F1: 0.72
threshold = 20, FPR: 0.241, TPR: 0.449, F1: 0.71
threshold = 30, FPR: 0.167, TPR: 0.4, F1: 0.7
threshold = 40, FPR: 0.12, TPR: 0.35, F1: 0.67
threshold = 50, FPR: 0.091, TPR: 0.298, F1: 0.62
threshold = 60, FPR: 0.074, TPR: 0.269, F1: 0.59
threshold = 70, FPR: 0.061, TPR: 0.242, F1: 0.55
threshold = 80, FPR: 0.052, TPR: 0.224, F1: 0.53
threshold = 90, FPR: 0.046, TPR: 0.21, F1: 0.51
threshold = 100, FPR: 0.041, TPR: 0.195, F1: 0.48
AUC for this ROC: 0.11898650000000001
```

We also find the F1 score for each threshold. The F1 score is a solid measure when both FPR and TPR need to be checked. In this case, we do not only want to rely on FPR to measure our performance because then, to have a "perfect" model, we could just classify everything as negative and never return a false positive. In contrast, we could also classify everything as positive if we only measured our TPR, because we would never return anything that wasn't positive, and every positive would be correctly identified.

The "best" F1 score we received was from control threshold = 10 distance points at an F1 of .72. For that reason, we believe this is the best threshold.