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

train_x = np.load("misuse_x_train.npy", allow_pickle=True)
train_y = np.load("misuse_y_train.npy", allow_pickle=True)
test_x = np.load("misuse_x_test.npy", allow_pickle=True)
test_y = np.load("misuse_y_test.npy", allow_pickle=True)

#BINARY CLASSIFICATION
#change existing numpy to binary. if ==0, leave it as 0. else change to 1
train_y_binary = np.where(train_y == 0,0,1)
test_y_binary = np.where(test_y == 0,0,1)

#train and test 10 models with varying k
classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier1.fit(train_x,train_y_binary)
pred1 = classifier1.predict(test_x)

classifier3 = KNeighborsClassifier(n_neighbors=3)
classifier3.fit(train_x,train_y_binary)
pred3 = classifier3.predict(test_x)

classifier5 = KNeighborsClassifier(n_neighbors=5)
classifier5.fit(train_x,train_y_binary)
pred5 = classifier5.predict(test_x)

classifier7 = KNeighborsClassifier(n_neighbors=7)
classifier7.fit(train_x,train_y_binary)
pred7 = classifier7.predict(test_x)

classifier9 = KNeighborsClassifier(n_neighbors=9)
classifier9.fit(train_x,train_y_binary)
pred9 = classifier9.predict(test_x)

classifier11 = KNeighborsClassifier(n_neighbors=11)
classifier11.fit(train_x,train_y_binary)
pred11 = classifier11.predict(test_x)

classifier13 = KNeighborsClassifier(n_neighbors=13)
classifier13.fit(train_x,train_y_binary)
pred13 = classifier13.predict(test_x)

classifier15 = KNeighborsClassifier(n_neighbors=15)
classifier15.fit(train_x,train_y_binary)
pred15 = classifier15.predict(test_x)

classifier17 = KNeighborsClassifier(n_neighbors=17)
classifier17.fit(train_x,train_y_binary)
pred17 = classifier17.predict(test_x)

classifier19 = KNeighborsClassifier(n_neighbors=19)
classifier19.fit(train_x,train_y_binary)
pred19 = classifier19.predict(test_x)

#binary performance measures
true_positive = []
false_positive = []
#print FPR and TPR
TP, FP, TN, FN = perf_measure(test_y_binary,pred1)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred1),2)
print("binary, k=1, FPR: " + str(FPR) + ", TPR: " + str(TPR) + ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred3)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred3),2)
print("binary, k=3, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred5)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred5),2)
print("binary, k=5, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred7)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred7),2)
print("binary, k=7, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred9)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred9),2)
print("binary, k=9, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred11)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred11),2)
print("binary, k=11, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred13)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred13),2)
print("binary, k=13, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred15)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred15),2)
print("binary, k=15, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred17)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred17),2)
print("binary, k=17, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

TP, FP, TN, FN = perf_measure(test_y_binary,pred19)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
f1 = round(f1_score(test_y_binary,pred19),2)
print("binary, k=19, FPR: " + str(FPR) + ", TPR: " + str(TPR)+ ", F1: " + str(f1))
true_positive.append(TPR)
false_positive.append(FPR)

true_positive = np.asarray(true_positive)
false_positive = np.asarray(false_positive)
#roc_coordinates = list(zip(false_positive,true_positive))
print("AUC for this ROC: " + str(auc(false_positive,true_positive)))

#plot ROC
plt.plot(false_positive,true_positive)
plt.title('ROC Curve for Binary Misuse Detection')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


#MULTICLASS CLASSIFICATION
#train and test 10 models with varying k
classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier1.fit(train_x,train_y)
pred1 = classifier1.predict(test_x)

classifier3 = KNeighborsClassifier(n_neighbors=3)
classifier3.fit(train_x,train_y)
pred3 = classifier3.predict(test_x)

classifier5 = KNeighborsClassifier(n_neighbors=5)
classifier5.fit(train_x,train_y)
pred5 = classifier5.predict(test_x)

classifier7 = KNeighborsClassifier(n_neighbors=7)
classifier7.fit(train_x,train_y)
pred7 = classifier7.predict(test_x)

classifier9 = KNeighborsClassifier(n_neighbors=9)
classifier9.fit(train_x,train_y)
pred9 = classifier9.predict(test_x)

classifier11 = KNeighborsClassifier(n_neighbors=11)
classifier11.fit(train_x,train_y)
pred11 = classifier11.predict(test_x)

classifier13 = KNeighborsClassifier(n_neighbors=13)
classifier13.fit(train_x,train_y)
pred13 = classifier13.predict(test_x)

classifier15 = KNeighborsClassifier(n_neighbors=15)
classifier15.fit(train_x,train_y)
pred15 = classifier15.predict(test_x)

classifier17 = KNeighborsClassifier(n_neighbors=17)
classifier17.fit(train_x,train_y)
pred17 = classifier17.predict(test_x)

classifier19 = KNeighborsClassifier(n_neighbors=19)
classifier19.fit(train_x,train_y)
pred19 = classifier19.predict(test_x)

#multiclass performance measures
#print FPR and TPR and confusion matrix
TP, FP, TN, FN = perf_measure(test_y,pred1)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=1, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred1))

TP, FP, TN, FN = perf_measure(test_y,pred3)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=3, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred3))

TP, FP, TN, FN = perf_measure(test_y,pred5)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=5, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred5))

TP, FP, TN, FN = perf_measure(test_y,pred7)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=7, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred7))

TP, FP, TN, FN = perf_measure(test_y,pred9)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=9, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred9))

TP, FP, TN, FN = perf_measure(test_y,pred11)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=11, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred11))

TP, FP, TN, FN = perf_measure(test_y,pred13)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=13, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred13))

TP, FP, TN, FN = perf_measure(test_y,pred15)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=15, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred15))

TP, FP, TN, FN = perf_measure(test_y,pred17)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=17, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred17))

TP, FP, TN, FN = perf_measure(test_y,pred19)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("multi, k=19, FPR: " + str(FPR) + ", TPR: " + str(TPR))
print(confusion_matrix(test_y,pred19))


