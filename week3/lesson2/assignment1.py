#!python3
"""Logistic regression"""

import pandas
import numpy as np
from scipy.spatial.distance import euclidean

from sklearn.metrics import roc_auc_score


def z(x, y, w):
    return 1.0 + np.exp(-y * x.dot(w))


def step_L2(x, y, w, k, C):
    l = len(x)
    return np.array([sum(y[i] * x[i][j] * (1 - 1 / z(x[i], y[i], w))
                         for i in range(l)) * k / l + w[j] - k * C * w[j]
                     for j in range(2)])


def step(x, y, w, k):
    l = len(x)
    return np.array([sum(y[i] * x[i][j] * (1.0 - 1.0 / z(x[i], y[i], w))
                         for i in range(l)) * k / l + w[j]
                     for j in range(2)])


def regression(x, y, w=np.array([0, 0]), e=1e-5, k=0.1):
    wn = np.array([-1000000, -1000000])
    cnt = 0
    while (cnt < 10000) and ((wn[0] == -1000000) or (euclidean(w, wn) > e)):
        if wn[0] != -1000000:
            w = wn
        wn = step(x, y, w, k)
        cnt += 1
    print('iterations with no regularization: {}'.format(cnt))
    return wn


def regression_L2(x, y, w=np.array([0, 0]), e=1e-5, k=0.1, C=10.0):
    wn = np.array([-1000000, -1000000])
    cnt = 0
    while (cnt < 10000) and ((wn[0] == -1000000) or (euclidean(w, wn) > e)):
        if wn[0] != -1000000:
            w = wn
        wn = step_L2(x, y, w, k, C)
        cnt += 1
    print('iterations with regularization: {}'.format(cnt))
    return wn


def a(x, w):
    return 1 / (1 + np.exp(-x.dot(w)))

data = pandas.read_csv('data-logistic.csv', header=None)
y = np.array(data[data.columns[0]])
x = np.array(data[data.columns[1:]])
w = regression(x, y)
w_L2 = regression_L2(x, y)
with open('ans.txt', 'w') as f:
    f.write('{:.3f} {:.3f}'.format(
        roc_auc_score(y, np.array([a(i, w) for i in x])),
        roc_auc_score(y, np.array([a(i, w_L2) for i in x]))
    ))
