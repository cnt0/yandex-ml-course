#!python3
"""Support Vector Machine"""

import numpy as np
import pandas

from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = TfidfVectorizer()
data = vectorizer.fit_transform(newsgroups.data, y=newsgroups.target)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(newsgroups.target.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(data, newsgroups.target)

clf2 = SVC(kernel='linear', random_state=241, C=gs.best_score_)
clf2.fit(data, newsgroups.target)

names = vectorizer.get_feature_names()
ans = [names[i] for i in
       np.array(clf2.coef_.indices)[np.argsort(np.abs(clf2.coef_.data))[-10:]]]

ans.sort()
with open('ans.txt', 'w') as f:
    f.write(' '.join(ans))
