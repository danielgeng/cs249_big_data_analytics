from ajw_process_data import *
import matplotlib.pyplot as plt

"""
:data: each row of data corresponds to a single user
:botlist: the row numbers of known bots
"""


def average_hist(data, botList):
    """
    Expects each row to be an array
    Plots the mean list for bots and humans
    :param data: rows of lists
    :param botList: the row numbers of known bots
    :return: plots the average list
    """

    botData = [0 for i in data[0]]
    humData = [0 for i in data[0]]
    for i, d in enumerate(data):
        if i in botList:
            botData = [x+y for (x,y) in zip(botData, d)]
        else:
            humData = [x+y for (x,y) in zip(humData, d)]

    botData = [float(d)/len(botList) for d in botData]
    humData = [float(d)/(len(data)-len(botList)) for d in humData]

    print botData
    print len(botList)

    print humData
    print len(data)

    plt.subplot(1,2,1)
    plt.plot(range(len(botData)), botData)
    plt.subplot(1,2,2)
    plt.plot(range(len(humData)), humData)
    plt.show()
    return

def heatmap(data1, data2, botList):
    return


def plot_feature_importance():
    return

def plot_AUC_over_n():
    return

def plot_best_ROC():
    return


def main():
    bidTuples, userList, botList = load_users()
    rts, auctionList, maxResponseTime = generate_rts(bidTuples, userList)
    rts_buckets = bucketize(rts, 5, outfile="../Data/rt_buckets.txt", maxVal=maxResponseTime)
    iats, maxIat = generate_iats(bidTuples, userList)
    iat_buckets = bucketize(iats, 5, outfile="../Data/iat_buckets.txt", maxVal=maxIat)
    average_hist(rts_buckets, botList)
    average_hist(iat_buckets, botList)

if __name__ == "__main__":
    main()