import random

import numpy as np
from ajw_process_data import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sys

def run_clf(clf, string=""):
    data, labels = new_features()
    # data, labels = five_and_rts()
    # data, labels = six_and_time_features()
    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)
    mean, AUC, tpr, fpr, threshs = kfoldcv(data, labels, clf, 100)
    # tpr = np.matrix(tpr)
    # fpr = np.matrix(fpr)
    # threshs = np.array(threshs)
    print string+":", str(mean)
    # print AUC
    # print tpr
    return mean, tpr, fpr


def select_best(clf, string="--"):
    n_range = (range(2,11)+[15,30,50,100,150,200])
    AUCs = []
    for n in n_range:
        meanAUC, tpr, fpr = run_clf(clf(n_estimators=n),string+" ("+str(n)+")")
        AUCs.append(meanAUC)
    sys.stdout.write("\n")
    plt.scatter(n_range,AUCs)
    plt.xlabel("n estimators")
    plt.ylabel("AUC")
    plt.title(string+" AUC vs. Number of Estimators")
    plt.savefig(string+"_AUC.png")
    plt.close()
    print string+" best n:", str(n_range[AUCs.index(max(AUCs))])
    return n_range[AUCs.index(max(AUCs))]


def run_combined():
    # ada = AdaBoostClassifier(n_estimators=9)
    # grad = GradientBoostingClassifier(n_estimators=50)
    # rforest = RandomForestClassifier(n_estimators=100)
    # bag = BaggingClassifier(n_estimators=200)
    # extra = ExtraTreesClassifier(n_estimators=150)
    grad1 = GradientBoostingClassifier(n_estimators=50)
    grad2 = GradientBoostingClassifier(n_estimators=50)
    grad3 = GradientBoostingClassifier(n_estimators=50)
    grad4 = GradientBoostingClassifier(n_estimators=50)
    grad5 = GradientBoostingClassifier(n_estimators=50)
    comb = CombinedClassifier([grad1, grad2, grad3,grad4,grad5])
    run_clf(comb, "Combined")


def run_adaboost():
    run_clf(AdaBoostClassifier(base_estimator=None, n_estimators=9, learning_rate=1, algorithm="SAMME.R", random_state=None),"Adaboost")
    # run_clf(GradientBoostingClassifier(n_estimators=9), "Gradient Boost")
    # run_clf(RandomForestClassifier(n_estimators=20), "Random Forest")
    # run_clf(BaggingClassifier(n_estimators=20), "Bagging")
    # run_clf(ExtraTreesClassifier(n_estimators=20), "Extra Trees")
    # fpr, tpr, thresholds = roc_curve(labels, estimate_scores[:,1], pos_label=1)
    # print roc_auc_score(labels, estimate_scores[:,1])
    # plt.plot(fpr, tpr)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Average ROC")
    # plt.show()
    return

def run_rforest():
    data, labels = six_features()
    rf = RandomForestClassifier


def run_importance(clf, feature_labels, string=""):
    # data, labels = six_and_time_features()
    # data = binned_time_features()
    # data, labels = five_and_rts()
    data, labels = new_features()
    feature_labels = ["","","","","",""]
    num_features = data.shape[1]
    print data.shape
    importances = [0]*num_features
    # dtree = DecisionTreeClassifier()
    dtree = GradientBoostingClassifier(n_estimators=30)
    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)

    for r in range(100):
        dtree.fit(data, labels)
        importances = [importances[i]+dtree.feature_importances_[i] for i in range(num_features)]
    importances = [importance/100 for importance in importances]

    non_zeros = [i for i in range(num_features) if not importances[i] == 0]
    importances = [importances[i] for i in non_zeros]
    feature_labels = [feature_labels[i] for i in non_zeros]
    bar_width = 0.7
    plt.bar(range(len(feature_labels)), importances, bar_width)
    plt.xticks([ind + +float(bar_width)/2 for ind in range(len(feature_labels))], feature_labels,rotation="vertical")
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.xlabel("Feature")
    plt.ylabel("Gini Importance")
    plt.title("Gini Importance v. Features for "+string+" Classifier")
    plt.show()


def run_decisiontree():
    data, labels = six_and_time_features()
    data = binned_time_features()
    # data, labels = five_and_rts()
    num_features = data.shape[1]
    print data.shape
    importances = [0]*num_features
    # dtree = DecisionTreeClassifier()
    dtree = GradientBoostingClassifier(n_estimators=30)
    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)

    for r in range(100):
        dtree.fit(data, labels)
        importances = [importances[i]+dtree.feature_importances_[i] for i in range(num_features)]
    importances = [importance/100 for importance in importances]

    six_labels = []#["meanIats", "meanRts", "bids", "bidsPerAuction", "numDevices", "numIps"]
    # six_labels = ["meanIats", "meanRts", "bids", "bidsPerAuction", "numDevices"]
    iat_labels = ["IAT bin "+str(k) for k in range(47)]
    rt_labels = ["RT bin "+str(k) for k in range(47)]
    all_labels = six_labels+iat_labels+rt_labels
    # all_labels = six_labels + rt_labels



    dtreeMean, dtreeAUC, dtreeTpr, dtreeFpr,dtreeThresh = kfoldcv(data, labels, dtree, 100)
    print dtreeMean
    print dtreeAUC
    print dtree.feature_importances_


def kfoldcv(data, labels, clf, k):
    AUCs = [0]*k
    ROC_tpr = [[] for i in range(k)]
    ROC_fpr = [[] for i in range(k)]
    ROC_thresh = [[] for i in range(k)]
    for i in range(k):
        # sys.stdout.write("|")
        AUCs[i], ROC_fpr[i], ROC_tpr[i], ROC_thresh[i] = cv(data, labels, clf)
    # sys.stdout.write("\n")

    return np.mean(np.array(AUCs)), np.array(AUCs), ROC_tpr, ROC_fpr, ROC_thresh

def cv(data, labels, clf):
    """
    Splits the data 80/20 and tests with AUC score and ROC curve
    :param data:
    :param labels:
    :param clf:
    :return:
    """
    # data and labels are both np.array type
    while True:
        shuffled = range(len(labels))
        random.shuffle(shuffled)
        twentyMark = len(labels)/5
        test_data = data[shuffled[:twentyMark],:]
        test_labels = labels[shuffled[:twentyMark]]
        train_data = data[shuffled[twentyMark:],:]
        train_labels = labels[shuffled[twentyMark:]]
        if sum(train_labels) > 20:
            break
    clf.fit(train_data, train_labels)
    estimate_scores = clf.predict_proba(test_data)
    fpr, tpr, thresholds = roc_curve(test_labels, estimate_scores[:,1], pos_label=1)
    return roc_auc_score(test_labels, estimate_scores[:,1]), fpr, tpr, thresholds


class CombinedClassifier:
    def __init__(self, clfList):
        self.clfList = clfList

    def fit(self, data, labels):
        for clf in self.clfList:
            clf.fit(data, labels)

    def predict_proba(self, test_data):
        probas = []
        for clf in self.clfList:
            probas.append(clf.predict_proba(test_data))
        # print  np.median(probas, axis=0)
        return np.median(probas, axis=0)


def main():
    # data = new_features(force=True)
    # bidTuples, userList, bostList = load_data()
    # ipEntropy = generate_IP_entropy(bidTuples, userList, force=False)
    # print ipEntropy
    # run_adaboost()
    # run_decisiontree()

    # for c in range(10):
    #     C = 3**(c-3)
    #     svm = SVC(kernel="rbf",probability=True, C=C)
    #     run_clf(svm,"RBF SVM (C="+str(C)+")")

    # # Single ROC for gradient boosting
    # mean, tpr, fpr = run_clf(GradientBoostingClassifier(n_estimators=50))
    # print tpr[0]
    # print fpr[0]
    # plt.plot(fpr[0], tpr[0])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Gradient Boosting Classifier ROC Curve")
    # plt.show()

    run_combined()
    # select_best(AdaBoostClassifier,"Adaboost")
    # select_best(GradientBoostingClassifier, "Gradient Boost")
    # select_best(RandomForestClassifier, "Random Forest")
    # select_best(BaggingClassifier, "Bagging")
    # select_best(ExtraTreesClassifier, "Extra Trees")
    return


if __name__ == "__main__":
    main()