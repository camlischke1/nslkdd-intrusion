import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, plot_roc_curve, auc, f1_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from matplotlib import pyplot as plt

#from https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i] and y_hat[i]!=0 and y_actual[i]!=0:
           TP += 1
        if y_hat[i]!=0 and y_actual[i]==0:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=0:
           FN += 1
    return(TP, FP, TN, FN)

train_x = np.load("anomaly_x_train.npy", allow_pickle=True)
test_x = np.load("anomaly_x_test.npy", allow_pickle=True)
test_y = np.load("anomaly_y_test.npy", allow_pickle=True)
train_y = np.zeros((train_x.shape[0]))

knn = NearestNeighbors(n_neighbors=1).fit(train_x)
knearest = KNeighborsClassifier(n_neighbors=1).fit(train_x,train_y)

#ten control thresholds, 10-100 respectively
pred1 = []
pred2 = []
pred3 = []
pred4 = []
pred5 = []
pred6 = []
pred7 = []
pred8 = []
pred9 = []
pred10 = []
for i in range(test_x.shape[0]):
    nearest = knn.kneighbors(test_x[i:i+1])
    nearest_distance = (nearest[0].ravel())[0]
    if nearest_distance > 10:
        pred1.append(1)
    else:
        pred1.append(0)
    if nearest_distance > 20:
        pred2.append(1)
    else:
        pred2.append(0)
    if nearest_distance > 30:
        pred3.append(1)
    else:
        pred3.append(0)
    if nearest_distance > 40:
        pred4.append(1)
    else:
        pred4.append(0)
    if nearest_distance > 50:
        pred5.append(1)
    else:
        pred5.append(0)
    if nearest_distance > 60:
        pred6.append(1)
    else:
        pred6.append(0)
    if nearest_distance > 70:
        pred7.append(1)
    else:
        pred7.append(0)
    if nearest_distance > 80:
        pred8.append(1)
    else:
        pred8.append(0)
    if nearest_distance > 90:
        pred9.append(1)
    else:
        pred9.append(0)
    if nearest_distance > 100:
        pred10.append(1)
    else:
        pred10.append(0)

pred1 = np.asarray(pred1)
pred2 = np.asarray(pred2)
pred3 = np.asarray(pred3)
pred4 = np.asarray(pred4)
pred5 = np.asarray(pred5)
pred6 = np.asarray(pred6)
pred7 = np.asarray(pred7)
pred8 = np.asarray(pred8)
pred9 = np.asarray(pred9)
pred10 = np.asarray(pred10)

true_positive = []
false_positive = []
#print FPR and TPR
TP, FP, TN, FN = perf_measure(test_y,pred1)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred1),2)
print("threshold = 10, FPR: " + str(FPR) + ", TPR: " + str(TPR) + ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred2)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred2),2)
print("threshold = 20, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred3)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred3),2)
print("threshold = 30, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred4)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred4),2)
print("threshold = 40, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred5)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred5),2)
print("threshold = 50, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred6)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred6),2)
print("threshold = 60, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred7)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred7),2)
print("threshold = 70, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred8)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred8),2)
print("threshold = 80, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred9)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred9),2)
print("threshold = 90, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y,pred10)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y,pred10),2)
print("threshold = 100, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

true_positive = np.asarray(true_positive)
false_positive = np.asarray(false_positive)
#roc_coordinates = list(zip(false_positive,true_positive))
print("AUC for this ROC: " + str(auc(false_positive,true_positive)))

#plot ROC
plt.plot(false_positive,true_positive)
plt.title('ROC Curve for Anomaly Detection')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()




