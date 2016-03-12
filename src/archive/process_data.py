__author__ = 'MsSaraMel'

import numpy as np
import csv
import math
import operator
import cPickle as pickle
import sys
import pandas as pd

TIME_LOG_BASE = 5

def load_bidder_data():
    #train_tbl = pd.read_csv("../Data/train_seq_bidder_id_time.csv")
    csvfile = open('../Data/train_seq_bidder_id_time.csv', 'rb')
    outfile = open('preproc_out.txt', 'w')
    train_tbl = csv.reader(csvfile, delimiter=',', escapechar='\\', quotechar=None)

    iat = {} # {user: [auction, time]}
    usermap = list()
    range_user = range(1,4000000,1) #user IDs will go from 1 to max with no gap
    counter = 0
    for toks in train_tbl:
        username, auction, time = [toks[0], toks[5], int(toks[8])]
        if username not in usermap:
            bidder = range_user[counter]
            usermap.append(username)
            counter += 1

        if bidder not in iat:
            iat[bidder] = []
        iat[bidder].append((auction, time))
    print >> outfile, '%s' %(bidder,)
    return iat, usermap


def process_data(iat, dataname):
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
    for user in iat:
        if len(iat[user]) <= 1: continue
        iat_counts = [0] * S
        cur_iat = sorted(iat[user], key=operator.itemgetter(1))
        for i in range(1, len(cur_iat)):
            time_diff = cur_iat[i][1] - cur_iat[i-1][1]
            iat_bucket = int(math.floor(math.log(1 + time_diff, TIME_LOG_BASE)))
            iat_counts[iat_bucket] += 1
        iat_arr.append(iat_counts)
        ids.append(user)

    with open('../data/%s_iat_bucketed.txt' % (dataname), 'w') as iat_file:
        for row in iat_arr:
            print >> iat_file, ' '.join([str(x) for x in row])

    iat_arr = np.array(iat_arr)
    return (iat_arr, ids)

def main():
    iat, usermap = load_bidder_data()
    process_data(iat, 'Facebook_bids')

if __name__=='__main__':
    main()
