import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

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

#train and test 10 models with varying k
classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier1.fit(train_x,train_y)
pred1 = classifier1.predict(test_x)
'''
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
'''
#print 10 confusion matrices to get FPR and TPR
TP, FP, TN, FN = perf_measure(test_y,pred1)
FPR = round(FP/test_y.shape[0],3)
TPR = round(TP/test_y.shape[0],3)
print("k=1, FPR: " + str(FPR) + ", TPR: " + str(TPR))