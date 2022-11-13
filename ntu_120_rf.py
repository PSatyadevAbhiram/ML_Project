import ntu_splits_ml as ntu_ml_d
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import tree


def fit_rand_forest(train, with_bootstrap=False, max_depth=5):
    rf_clf = RandomForestClassifier()
    rf_clf.fit(*train)
    return rf_clf


def test_rand_forest(rf_clf, test):
    gt = np.array(test[1], dtype=int)
    pred = np.array(rf_clf.predict(test[0]), dtype=int)
    total_samples = gt.shape[0]
    correct = (np.sum(gt==pred))
    acc = round(correct/total_samples, 3)
    print(f"The accuracy of the random forest classifier is {acc}")


if __name__ == "__main__":
    train, test = ntu_ml_d.get_ntu_120_xsub_train(tp=False), ntu_ml_d.get_ntu_120_xsub_test(tp=False)
    rf_clf = fit_rand_forest(train)
    test_rand_forest(rf_clf, test)
