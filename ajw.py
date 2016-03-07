import numpy as np
from ajw_process_data import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def run_adaboost():
    data, labels = six_features()
    ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1, algorithm="SAMME.R", random_state=None)
    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)
    ada.fit(data, labels)

    # TODO (ajw): cross validate. This is just to get a feel for how the function will run
    estimate_scores = ada.predict_proba(data)
    print labels
    print estimate_scores
    p, r, t = precision_recall_curve(labels, estimate_scores[:,1])
    print roc_auc_score(labels, estimate_scores[:,1])
    plt.scatter(r, p)
    plt.show()
    return


def main():
    run_adaboost()
    return


if __name__ == "__main__":
    main()