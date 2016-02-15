__author__ = 'MsSaraMel'

import numpy as np
import csv
import sys
import pandas as pd

#with open('../Data/bids.csv', 'r') as f:
#    reader = csv.reader(f);
#    data = [data for data in reader];
#data_array = np.asarray(data);


bidder = pd.read_csv("../Data/train.csv")
bids = pd.read_csv("../Data/bids.csv")
bids = bids.dropna(axis=1)
merged = bidder.merge(bids, on='bidder_id')
merged.to_csv("../Data/join_train.csv", index=False)


print "finished"
