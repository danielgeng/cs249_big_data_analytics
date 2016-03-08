import random

import numpy as np
from ajw_process_data import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def run_adaboost():
    data, labels = six_features()
    ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1, algorithm="SAMME.R", random_state=None)
    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)
    ada.fit(data, labels)

    # TODO (ajw): cross validate. This is just to get a feel for how the function will run
    adaMean, adaAUC = kfoldcv(data, labels, ada, 100)
    print adaMean
    print adaAUC

    # fpr, tpr, thresholds = roc_curve(labels, estimate_scores[:,1], pos_label=1)
    # print roc_auc_score(labels, estimate_scores[:,1])
    # plt.scatter(fpr, tpr)
    # plt.show()
    return


def run_decisiontree():
    data, labels = six_features()
    dtree = DecisionTreeClassifier()
    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)
    dtree.fit(data, labels)
    print dtree.feature_importances_


def kfoldcv(data, labels, clf, k):
    AUCs = [0]*k
    for i in range(k):
        AUCs[i] = cv(data, labels, clf)
    return np.mean(np.array(AUCs)), np.array(AUCs)

def cv(data, labels, clf):
    # data and labels are both np.array type
    shuffled = range(len(labels))
    random.shuffle(shuffled)
    twentyMark = len(labels)/5
    test_data = data[shuffled[:twentyMark],:]
    test_labels = labels[shuffled[:twentyMark]]
    train_data = data[shuffled[twentyMark:],:]
    train_labels = labels[shuffled[twentyMark:]]
    clf.fit(train_data, train_labels)
    estimate_scores = clf.predict_proba(test_data)
    return roc_auc_score(test_labels, estimate_scores[:,1])


def main():
    run_adaboost()
    # run_decisiontree()
    return


if __name__ == "__main__":
    main()