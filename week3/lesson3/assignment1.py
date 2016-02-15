#!/usr/bin/env python3.5
"""Classification quality metrics"""


import numpy as np
import pandas
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    f1_score, roc_auc_score, precision_recall_curve


data = pandas.read_csv('classification.csv')

y_true = data['true']
y_pred = data['pred']

with open('ans1.txt', 'w') as f:
    f.write('{} {} {} {}'.format(
        data[(y_true == 1) & (y_pred == 1)].count()[0],
        data[(y_true == 0) & (y_pred == 1)].count()[0],
        data[(y_true == 1) & (y_pred == 0)].count()[0],
        data[(y_true == 0) & (y_pred == 0)].count()[0],
    ))

with open('ans2.txt', 'w') as f0:
    f0.write('{:.2f} {:.2f} {:.2f} {:.2f}'.format(
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    ))


scores = pandas.read_csv('scores.csv')
true = scores['true']
cols = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']
pred = scores[cols]

with open('ans3.txt', 'w') as f:
    f.write('{}'.format(
        cols[np.argmax([roc_auc_score(true, pred[x]) for x in cols])])
    )


def prc_m(y_true, y_pred):
    z = precision_recall_curve(y_true, y_pred)
    return z[0][z[1] >= 0.7].max()

with open('ans4.txt', 'w') as f:
    f.write('{}'.format(
        cols[np.argmax([prc_m(true, pred[x]) for x in cols])])
    )
