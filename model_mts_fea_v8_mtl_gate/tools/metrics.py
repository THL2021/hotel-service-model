"""for special metrics"""
import numpy as np
from sklearn.metrics import roc_auc_score


# rank metrics
def dcg_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(y_true, y_score):
    scores = []
    # Iterate over each y_value_true and compute the DCG score
    for y_value_true, y_value_score in zip(y_true, y_score):
        actual = dcg_score(y_value_true, y_value_score)
        best = dcg_score(y_value_true, y_value_true)
        # print(best)
        scores.append(actual / best)
    return np.mean(scores)


def gauc_and_ndcg(preds_dict, labels_dict):
    y_true = []
    y_score = []
    group_aucs = []
    for key in preds_dict:
        preds_ = preds_dict[key]
        labels_ = labels_dict[key]
        if len(labels_) == sum(labels_) or sum(labels_) == 0:
            pass
        else:
            group_aucs.append(roc_auc_score(labels_, preds_))
            y_true.append(labels_)
            y_score.append(preds_)
    ndcg = ndcg_score(y_true, y_score)
    gauc = np.mean(group_aucs)
    count = len(y_true)
    return gauc, ndcg, count


def seg_auc(preds_dict, labels_dict):
    seg_aucs = {}
    for key in preds_dict:
        preds_ = preds_dict[key]
        labels_ = labels_dict[key]
        seg_aucs[key] = (roc_auc_score(labels_, preds_), len(labels_))
    return seg_aucs
