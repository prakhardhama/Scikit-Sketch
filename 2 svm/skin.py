import numpy as np
from os import path
from sklearn import svm
import matplotlib.pyplot as plt

file_path = path.relpath("Skin_Nonskin.txt");
f = open(file_path)
f.readline()

skin_arr = np.loadtxt(f)
f.close()

X = skin_arr[:, 0:3]
y = skin_arr[:, 3]

n_sample = len(X)
np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order].astype(np.int)
y = y[order].astype(np.int)

#X = X[0:10000]
#y = y[0:10000]
#n_sample = 10000
n_train = int(.9 * n_sample)
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]
n_test = len(y_test)

clf = svm.SVC(gamma=.001, C=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_dist = clf.decision_function(X_test)
#score = clf.score(X_test, y_test)

c1_a = 0
c1_p = 0
c2_a = 0
c2_p = 0

for i in range(n_test):
    if y_test[i] == 1:
        c1_a += 1
        if y_pred[i] == 1:
            c1_p += 1
    else:
        c2_a +=1
        if y_pred[i] == 2:
            c2_p += 1

fo = open("skin_RESULT.txt", "w")

fo.write("Overall Accuracy: "+str((c1_p+c2_p)*100.0/n_test)+"\n")
if c1_a !=0:
    fo.write("Class 1 Accuracy: "+str(c1_p*100.0/c1_a)+"\n")
if c2_a !=0:
    fo.write("Class 2 Accuracy: "+str(c2_p*100.0/c2_a)+"\n")

fo.write("Predict\tActual\tDist. from plane\n")
for index in range(n_test):
   fo.write(str(y_pred[index])+"\t"+str(y_test[index])+"\t"+"{:.5f}".format(y_dist[index])+"\n")

fo.close()
