from collections import Counter

import numpy as np
from scipy import stats


def cal_acc(preds, goldens):
    assert len(preds) == len(goldens), f"{len(preds)} is not equal to {len(goldens)}"
    correct = 0.0
    for x, y in zip(preds, goldens):
        if x == y:
            correct += 1
    return correct / (len(goldens) + 1e-9)


def cal_precision(preds, goldens):
    tp = len(set(preds) & set(goldens))
    return tp / (len(preds) + 1e-9)


def cal_recall(preds, goldens):
    tp = len(set(preds) & set(goldens))
    return tp / (len(goldens) + 1e-9)


def cal_PRF1(preds, goldens):
    precision = cal_precision(preds, goldens)
    recall = cal_recall(preds, goldens)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return precision, recall, f1


def cal_PRF1_average(prf1s):
    """
    计算平均
    input: list of (p, r, f1)
    """
    ave_pre = sum([i[0] for i in prf1s]) / len(prf1s)
    ave_rec = sum([i[1] for i in prf1s]) / len(prf1s)
    ave_f1 = sum([i[2] for i in prf1s]) / len(prf1s)
    return (
        ave_pre,
        ave_rec,
        ave_f1,
    )


def statistic_info(nums=[1, 1, 2, 3], _round: int = None):
    """
    mean, max, min, mode, median, 25%quantile, 75%quantile, 90%quantile, 95%quantile, standard deviation, skewness, kurtosis
    """
    nums = np.array(nums)
    _max = nums.max()
    _min = nums.min()
    _mean = nums.mean()
    _median = np.median(nums)
    _most = stats.mode(nums)[0]
    _percent_25 = np.percentile(nums, 25)
    _percent_75 = np.percentile(nums, 75)
    _percent_90 = np.percentile(nums, 90)
    _percent_95 = np.percentile(nums, 95)
    _percent_99 = np.percentile(nums, 99)
    _std = np.std(nums)
    _skew = stats.skew(nums)  # 偏度
    _kurtosis = stats.kurtosis(nums)  # 峰度

    return {
        "mean": float(_mean) if _round is None else round(float(_mean), int(_round)),
        "max": float(_max) if _round is None else round(float(_max), int(_round)),
        "min": float(_min) if _round is None else round(float(_min), int(_round)),
        "mode": float(_most) if _round is None else round(float(_most), int(_round)),
        "median": float(_median) if _round is None else round(float(_median), int(_round)),
        "25%quantile": float(_percent_25) if _round is None else round(float(_percent_25), int(_round)),
        "75%quantile": float(_percent_75) if _round is None else round(float(_percent_75), int(_round)),
        "90%quantile": float(_percent_90) if _round is None else round(float(_percent_90), int(_round)),
        "95%quantile": float(_percent_95) if _round is None else round(float(_percent_95), int(_round)),
        "99%quantile": float(_percent_99) if _round is None else round(float(_percent_99), int(_round)),
        "standard deviation": float(_std) if _round is None else round(float(_std), int(_round)),
        "skewness": float(_skew) if _round is None else round(float(_skew), int(_round)),
        "kurtosis": float(_kurtosis) if _round is None else round(float(_kurtosis), int(_round)),
    }


def cal_freq_dist(lens, orderby="most_common"):
    """
    alculate cumulative distribution
    orderby:
        1. most_common: frequency
        2. number_line: number line
    """
    assert orderby in [
        "most_common",
        "number_line",
    ], f"orderby: {orderby} not in [`most_common`,`number_line`]"
    if not lens:
        return {}
    total = len(lens)
    c = Counter(lens)

    if orderby == "most_common":
        c = c.most_common()
    else:
        try:
            c = sorted(c.items(), key=lambda x: int(x[0]))
        except:
            c = sorted(c.items(), key=lambda x: x[0])

    cum = 0
    cum_dict = {}
    for i in c:
        v, freq = i
        cum += freq
        cum_dict[v] = float(cum) / total
    return cum_dict
