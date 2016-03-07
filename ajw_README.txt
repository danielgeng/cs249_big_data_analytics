### README for Andrew's code

# ajw_process_data.py
This file contains the functions I use to get some features from the data files.

Some functions have a :force: input parameter. If this is true, the function will go through the entire computation. Otherwise, it will try to load try to load data from each variable's corresponding pickle file.

To run, set DATAHOME to be the path to the directory containing train_seq_bidder_id_time.csv.

To generate features with ajw_process_data.py:
	load_users()
		bidTuples:
		botList:
		userLst:

	generate_mean_iats()
	generate_iats()
		userData:

	generate_mean_rts()
	generate_rts()
		auctionData:
		auctionList:

	generate_bid_counts()
		userAuctionData:
		userBidCounts:
		userBidCountsPerAuction

	six_features()
		six_features: [meanIats, meanRts, bids, bidsPerAuction, numDevices, numIps]

	generate_num_devices()

	generate_num_ips

	bucketize()