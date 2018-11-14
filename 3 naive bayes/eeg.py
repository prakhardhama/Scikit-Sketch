# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 00:02:01 2016

@author: Prakhar Dhama
"""

import numpy as np
from os import path
from sklearn.naive_bayes import GaussianNB

file_path = path.relpath("EEG Eye State.arff");
eeg_arr = np.genfromtxt(file_path,skip_header=19,delimiter=',')

X = eeg_arr[:, 0:14]
y = eeg_arr[:, 14]

n_sample = len(X)
np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.int)
#
#X = X[0:100]
#y = y[0:100]
#n_sample = 100
n_train = int(.9 * n_sample)
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]
n_test = len(y_test)

clf = GaussianNB()
clf.fit(X_train, y_train)
#clf.partial_fit(X_train, y_train,np.unique(y_train))

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

c1_a = 0
c1_p = 0
c2_a = 0
c2_p = 0

for i in range(n_test):
    if y_test[i] == 0:
        c1_a += 1
        if y_pred[i] == 0:
            c1_p += 1
    else:
        c2_a +=1
        if y_pred[i] == 1:
            c2_p += 1

fo = open("eeg_RESULT.txt", "w")

fo.write("Overall Accuracy: "+str((c1_p+c2_p)*100.0/n_test)+"\n")
if c1_a !=0:
    fo.write("Class 0 Accuracy: "+str(c1_p*100.0/c1_a)+"\n")
if c2_a !=0:
    fo.write("Class 1 Accuracy: "+str(c2_p*100.0/c2_a)+"\n")
fo.write("Predict\tActual\tProbability\n")
for index in range(n_test):
   fo.write(str(y_pred[index])+"\t"+str(y_test[index])+"\t["+"{:.4f}".format(y_prob[index][0])+"  {:.4f}".format(y_prob[index][1])+"]\n")
   
fo.close()
