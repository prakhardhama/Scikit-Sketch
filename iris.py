# -*- coding: utf-8 -*-
"""
Created on Thu Mar  31 00:11:52 2016

@author: Prakhar Dhama
"""

import numpy as np
from sklearn import preprocessing
from sklearn.lda import LDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = np.loadtxt('iris.data',
                  delimiter=',',
                  dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'iris class'),
                         'formats': (np.float, np.float, np.float, np.float, '|S15')})
#X = iris[['sepal length', 'sepal width', 'petal length', 'petal width']]
iris_class = iris['iris class'].astype(str)

le = preprocessing.LabelEncoder()
le.fit(iris_class)
y = le.transform(iris_class)

iris_label = ['setosa', 'versic', 'virginica']

ret = [[0 for x in range(150)] for x in range(4)] 
ret[0]= iris['sepal length']
ret[1]= iris['sepal width']
ret[2]= iris['petal length']
ret[3]= iris['petal width']

X = [[0 for x in range(4)] for x in range(150)] 

for i in range(4): 
    for j in range(150): 
        X[j][i] = ret[i][j];
    


lda = LDA(n_components = 2)
X_r = lda.fit(X, y).transform(X)

plt.figure()
for c, i, l, m in zip("rgb", [0, 1, 2], iris_label, ['3', '.', '4']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=l, marker=m)
plt.legend()
plt.title('LDA of IRIS dataset')

pca = PCA(n_components = 2)
X_r2 = pca.fit(X).transform(X)

plt.figure()
for c, i, l, m in zip("rgb", [0, 1, 2], iris_label, ['3', '.', '4']):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=l, marker=m)
plt.legend()
plt.title('PCA of IRIS dataset')

plt.show()