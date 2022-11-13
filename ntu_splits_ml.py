import numpy as np
import pickle

# Preparing features for sci-kit learn algorithms
# X vals are of the form (n_samples, n_features), Y vals (labels)
# are of the form (n_samples)


xset_train_fps_txt = "./xset_train.txt"
xset_test_fps_txt = "./xset_val.txt"
xsub_train_fps_txt = "./xsub_train.txt"
xsub_test_fps_txt = "./xsub_val.txt"
feats_d_fp = "./ntu120_ml_feats.pkl" # no pca
feats_d_fp_pca = "../Data/NTU_RGBD/ntu120_ml_feats_with_pca.pkl"


def filter_d(d, fns, tp=True):
    filtered_d = {key: d[key] for key in fns}
    for key in list(filtered_d.keys()):
        if filtered_d[key]["num_people"] == 2 and not tp:
            del filtered_d[key]
    return filtered_d


def get_train_tuple(fps_txt_fp, tp=True):
    with open(fps_txt_fp, "r") as f:
        lines = f.readlines()
    stripped_lines = []
    for l in lines:
        stripped_lines.append(l.strip("\n").strip("\r"))
    with open(feats_d_fp_pca, "rb") as f:
        feats_d = pickle.load(f)
    f_feats_d = filter_d(feats_d, stripped_lines, tp)
    X = [f_feats_d[key]["features"] for key in f_feats_d.keys()]
    Y = [f_feats_d[key]["class"] for key in f_feats_d.keys()]
    return X, Y


def get_ntu_120_xsub_train(tp=True):
    return get_train_tuple(xsub_train_fps_txt, tp=tp)


def get_ntu_120_xsub_test(tp=True):
    return get_train_tuple(xsub_test_fps_txt, tp=tp)


def get_ntu_120_xset_train(tp=True):
    return get_train_tuple(xset_train_fps_txt, tp=tp)


def get_ntu_120_xset_test(tp=True):
    return get_train_tuple(xset_test_fps_txt, tp=tp)