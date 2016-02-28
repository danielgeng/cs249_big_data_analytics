#!/usr/bin/env python

import numpy as np
import csv
import sys
import pandas as pd

bidder = pd.read_csv("~/Downloads/train.csv")
bids = pd.read_csv("~/Downloads/bids.csv")
merged = bidder.merge(bids, on='bidder_id')
merged.to_csv("data/join_train.csv", index=False)

# schema: bidder_id,payment_account,address,outcome,bid_id,auction,merchandise,device,time,country,ip,url
