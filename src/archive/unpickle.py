#!/usr/bin/env python

import csv
import cPickle as pickle

(iat_arr, complete_iat_arr, rsp_arr, complete_rsp_arr, ids, usermap) = pickle.load(open("data/FacebookBids_2bucketed.pickle"))
with open('data/_train_iat_arr.csv', 'w') as file:
	for row in iat_arr:
		print >> file, ','.join([str(x) for x in row])

with open('data/_train_complete_iat_arr.csv', 'w') as file:
	for row in complete_iat_arr:
		print >> file, ','.join([str(x) for x in row])

with open('data/_train_rsp_arr.csv', 'w') as file:
	for row in rsp_arr:
		print >> file, ','.join([str(x) for x in row])

with open('data/_train_complete_rsp_arr.csv', 'w') as file:
	for row in complete_rsp_arr:
		print >> file, ','.join([str(x) for x in row])

with open('data/_train_ids', 'w') as file:
	for row in ids:
		print >> file, row

with open('data/_train_usermap.csv', 'w') as file:
	for row in usermap:
		print >> file, ','.join([str(x) for x in row])
