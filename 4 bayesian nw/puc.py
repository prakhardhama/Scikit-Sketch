# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 00:51:02 2016

@author: Prakhar Dhama
"""

import numpy as np
from os import path
from sklearn.naive_bayes import BernoulliNB

file_path = path.relpath("puc.csv");
puc_arr = np.genfromtxt(file_path,skip_header=1,delimiter=';')
puc_arr = np.delete(puc_arr, 0, 1);

X = puc_arr[:, 0:17]
y = puc_arr[:, 17]

"""
woman=0 man=1
sitting=0 sittingdown=1 standing=2 stadingup=3 walking=4
"""

n_sample = len(X)
np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order].astype(np.int)
y = y[order].astype(np.int)

n_train = int(.9 * n_sample)
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]
n_test = len(y_test)

clf = BernoulliNB()
clf.fit(X_train, y_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)


c0_a = 0
c0_p = 0
c1_a = 0
c1_p = 0
c2_a = 0
c2_p = 0
c3_a = 0
c3_p = 0
c4_a = 0
c4_p = 0

for i in range(n_test):
    if y_test[i] == 0:
        c0_a += 1
        if y_pred[i] == 0:
            c0_p += 1
    elif y_test[i]==1:
        c1_a +=1
        if y_pred[i] == 1:
            c1_p += 1
    elif y_test[i]==2:
        c2_a +=1
        if y_pred[i] == 2:
            c2_p += 1
    elif y_test[i]==3:
        c3_a +=1
        if y_pred[i] == 3:
            c3_p += 1
    else:
        c4_a +=1
        if y_pred[i] == 4:
            c4_p += 1
            
fo = open("puc_RESULT.txt", "w")

fo.write("Overall Accuracy: "+str((c0_p+c1_p+c2_p+c3_p+c4_p)*100.0/n_test)+"\n")
if c1_a !=0:
    fo.write("Class 0 Accuracy: "+str(c0_p*100.0/c0_a)+"\n")
if c1_a !=0:
    fo.write("Class 1 Accuracy: "+str(c1_p*100.0/c1_a)+"\n")
if c2_a !=0:
    fo.write("Class 2 Accuracy: "+str(c2_p*100.0/c2_a)+"\n")
if c3_a !=0:
    fo.write("Class 3 Accuracy: "+str(c3_p*100.0/c3_a)+"\n")
if c4_a !=0:
    fo.write("Class 4 Accuracy: "+str(c4_p*100.0/c4_a)+"\n")

fo.write("Predict\tActual\n")
for index in range(n_test):
   fo.write(str(y_pred[index])+"\t"+str(y_test[index])+"\n")

fo.close()
    