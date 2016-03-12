#!/usr/bin/env python

from sklearn import *
import numpy as np
import csv
import warnings
import time

np.set_printoptions(suppress=True)

start_time = time.time()

features = np.genfromtxt('data/_train_complete_iat_arr.csv', delimiter = ',')
labels = np.genfromtxt('data/_train_usermap.csv', delimiter = ',', usecols = (1))

print(features.shape, labels.shape)

ls = svm.SVC()
ls.fit(features, labels)
print(ls)

print(np.mean(labels))

cv = cross_validation.ShuffleSplit(features.shape[0])
accus = cross_validation.cross_val_score(estimator=ls, X=features, y=labels, cv=cv)
print "accuracies %s" % (accus)
print "average accuracy %s" % (np.mean(accus))

print "time taken: %s seconds" % (time.time() - start_time)
