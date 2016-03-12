#!/usr/bin/env python

from sklearn import *
import numpy as np
import csv
import warnings
import time

np.set_printoptions(suppress=True)

start_time = time.time()

features1 = np.genfromtxt('data/_train_complete_iat_arr.csv', delimiter = ',')
features2 = np.genfromtxt('data/_train_complete_rsp_arr.csv', delimiter = ',')
features = np.hstack((features1, features2))
labels = np.genfromtxt('data/_train_usermap.csv', delimiter = ',', usecols = (1))

print(features.shape, labels.shape)

for i in range(1, 12):
	accus = []
	for j in range(0, 10):
		clf = cluster.KMeans(n_clusters=i)
		clf.fit(features, labels)
		pred = clf.predict(features)	
		accus.append(metrics.accuracy_score(labels, pred))
		# print metrics.roc_curve(labels, pred)
	print "clusters: %s; accuracy: %s" % (i, np.mean(accus))
	print accus

print "time taken: %s seconds" % (time.time() - start_time)
