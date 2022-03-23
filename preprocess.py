import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

train = np.asarray(pd.read_csv('20PercentTrainingSet.csv',header=None))
test = np.asarray(pd.read_csv('KDDTest+.csv',header=None))

#get rid of nominal attributes
train = np.delete(train,1, axis=1)
train = np.delete(train,1, axis=1)
train = np.delete(train,1, axis=1)
test = np.delete(test,1, axis=1)
test = np.delete(test,1, axis=1)
test = np.delete(test,1, axis=1)

#split x and y
x_train = train[:,:38]
x_test = test[:,:38]
y_train = train[:,38]
y_test = test[:,38]

#transform nominal class labels into numerical
misuse_y_train = []
for i in range(y_train.shape[0]):
    if y_train[i] == 'normal':
        misuse_y_train.append(0)
    elif y_train[i] == 'back' or y_train[i] == 'land' or y_train[i] == 'neptune' or y_train[i] == 'pod' or y_train[i] == 'smurf' or y_train[i] == 'teardrop' or y_train[i] == 'apache2' or y_train[i] == 'udpstorm' or y_train[i] == 'mailbomb' or y_train[i] == 'processtable' or y_train[i] == 'worm':
        misuse_y_train.append(1)
    elif y_train[i] == 'satan' or y_train[i] == 'ipsweep' or y_train[i] == 'nmap' or y_train[i] == 'portsweep' or y_train[i] == 'mscan' or y_train[i] == 'saint':
        misuse_y_train.append(2)
    elif y_train[i] == 'guess_passwd' or y_train[i] == 'ftp_write' or y_train[i] == 'imap' or y_train[i] == 'phf' or y_train[i] == 'multihop' or y_train[i] == 'warezmaster' or y_train[i] == 'warezclient' or y_train[i] == 'spy' or y_train[i] == 'xlock' or y_train[i] == 'xsnoop' or y_train[i] == 'snmpguess' or y_train[i] == 'snmpgetattack' or y_train[i] == 'httptunnel' or y_train[i] == 'sendmail' or y_train[i] == 'named':
        misuse_y_train.append(3)
    elif y_train[i] == 'buffer_overflow' or y_train[i] == 'loadmodule' or y_train[i] == 'rootkit' or y_train[i] == 'perl' or y_train[i] == 'sqlattack' or y_train[i] == 'xterm' or y_train[i] == 'ps':
        misuse_y_train.append(4)
misuse_y_train = np.asarray(misuse_y_train)

misuse_y_test = []
for i in range(y_test.shape[0]):
    if y_test[i] == 'normal':
        misuse_y_test.append(0)
    elif y_test[i] == 'back' or y_test[i] == 'land' or y_test[i] == 'neptune' or y_test[i] == 'pod' or y_test[i] == 'smurf' or y_test[i] == 'teardrop' or y_test[i] == 'apache2' or y_test[i] == 'udpstorm' or y_test[i] == 'mailbomb' or y_test[i] == 'processtable' or y_test[i] == 'worm':
        misuse_y_test.append(1)
    elif y_test[i] == 'satan' or y_test[i] == 'ipsweep' or y_test[i] == 'nmap' or y_test[i] == 'portsweep' or y_test[i] == 'mscan' or y_test[i] == 'saint':
        misuse_y_test.append(2)
    elif y_test[i] == 'guess_passwd' or y_test[i] == 'ftp_write' or y_test[i] == 'imap' or y_test[i] == 'phf' or y_test[i] == 'multihop' or y_test[i] == 'warezmaster' or y_test[i] == 'warezclient' or y_test[i] == 'spy' or y_test[i] == 'xlock' or y_test[i] == 'xsnoop' or y_test[i] == 'snmpguess' or y_test[i] == 'snmpgetattack' or y_test[i] == 'httptunnel' or y_test[i] == 'sendmail' or y_test[i] == 'named':
        misuse_y_test.append(3)
    elif y_test[i] == 'buffer_overflow' or y_test[i] == 'loadmodule' or y_test[i] == 'rootkit' or y_test[i] == 'perl' or y_test[i] == 'sqlattack' or y_test[i] == 'xterm' or y_test[i] == 'ps':
        misuse_y_test.append(4)
misuse_y_test = np.asarray(misuse_y_test)

np.save('misuse_x_train.npy',x_train)
np.save('misuse_x_test.npy',x_test)
np.save('misuse_y_test.npy',misuse_y_test)
np.save('misuse_y_train.npy',misuse_y_train)

#get only normal samples for anomaly_x_train
anomaly_x_train = []
for i in range(x_train.shape[0]):
    if misuse_y_train[i] == 0:
        anomaly_x_train.append(x_train[i])
anomaly_x_train = np.asarray(anomaly_x_train)

#change all labels to binary
anomaly_y_test = np.where(misuse_y_test != 0,1,0)

#y_train is all zeroes for anomaly
np.save('anomaly_x_train.npy', anomaly_x_train)
np.save('anomaly_x_test.npy', x_test)
np.save('anomaly_y_test.npy', anomaly_y_test)

print(anomaly_x_train.shape)
print(x_test.shape)
print(anomaly_y_test.shape)

print()
print(x_train.shape)
print(misuse_y_train.shape)
print(x_test.shape)
print(misuse_y_test.shape)
