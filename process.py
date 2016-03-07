#!/usr/bin/env python

import numpy as np
import csv
import math
import operator
import cPickle as pickle
import sys
import pandas as pd

TIME_LOG_BASE = 5

def load_bidder_data():
    csvfile = open('data/join_train.csv', 'rb')
    train_tbl = csv.reader(csvfile, delimiter=',', escapechar='\\', quotechar=None)

    iat = {} # {user: [auction, time]}
    otherdata = [[]]
    usermap = list()
    counter = 0
    for toks in train_tbl:
        username, auction, time = [toks[0], toks[5], int(toks[8])]
        if username not in usermap:
            bidder = counter
            usermap.append(username)
            tmp = [toks[0], toks[3], toks[9]] # add other attributes if needed (current is bidder_id, outcome, country); ip needs to be another list for each user
            otherdata.append(tmp)
            counter += 1
        
        bidder = usermap.index(username)

        if bidder not in iat:
            iat[bidder] = []
        iat[bidder].append((auction, time))
    return iat, otherdata


def process_data(iat, otherdata, dataname):
#Method based on BIRDNEST source code
    iat_arr = []
    ids = []
    max_time_diff = -1
    for user in iat:
        cur_iat = sorted(iat[user], key=operator.itemgetter(1))
        for i in range(1, len(cur_iat)):
            time_diff = cur_iat[i][1] - cur_iat[i-1][1]
            max_time_diff = max(max_time_diff, time_diff)

    S = int(1 + math.floor(math.log(1 + max_time_diff, TIME_LOG_BASE)))
    ctr = 0
    for user in iat:
        if len(iat[user]) <= 1:
            otherdata.remove(otherdata[ctr])
            continue
        iat_counts = [0] * S
        cur_iat = sorted(iat[user], key=operator.itemgetter(1))
        for i in range(1, len(cur_iat)):
            time_diff = cur_iat[i][1] - cur_iat[i-1][1]
            iat_bucket = int(math.floor(math.log(1 + time_diff, TIME_LOG_BASE)))
            iat_counts[iat_bucket] += 1
        iat_arr.append(iat_counts)
        ids.append(user)
        ctr += 1

    with open('data/%s_iat_bucketed.csv' % (dataname), 'w') as iat_file:
        for row in iat_arr:
            print >> iat_file, ','.join([str(x) for x in row])

    with open('data/%s_other_data.csv' % (dataname), 'w') as otherfile:
        for row in otherdata:
            print >> otherfile, ','.join([str(x) for x in row])

    iat_arr = np.array(iat_arr)
    return (iat_arr, ids)

def main():
    iat, otherdata = load_bidder_data()
    process_data(iat, otherdata, 'train')

if __name__=='__main__':
    main()
