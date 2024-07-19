import random
from typing import List

"""
Modified from the original WebQSP evaluation script.
random Hist@1: https://github.com/microsoft/KC/blob/main/papers/TIARA/src/utils/statistics/webqsp_legacy_eval.py
"""


def FindInList(entry, elist):
    for item in elist:
        if entry == item:
            return True
    return False


def CalculatePRF1(goldAnswerList: List[str], predAnswerList: List[str]):
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return [
                1.0,
                1.0,
                1.0,
                1.0,
            ]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
        else:
            return [
                0.0,
                1.0,
                0.0,
                0.0,
            ]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
    elif len(predAnswerList) == 0:
        return [
            1.0,
            0.0,
            0.0,
            0.0,
        ]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
    else:
        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0
        # Calculate true positives (tp) and false negatives (fn) directly for each element in the predicted list and the golden list.
        for gentry in goldAnswerList:
            if FindInList(gentry, predAnswerList):  # Calculate how many are correct in the plist.
                tp += 1
            else:
                fn += 1
        for pentry in predAnswerList:
            if not FindInList(pentry, goldAnswerList):  # Calculate how many are wrong in the glist.
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = (2 * precision * recall) / (precision + recall)

        num_random = 100
        random_hit = 0
        for i in range(num_random):
            random_ans = random.choice(predAnswerList)
            if random_ans in goldAnswerList:
                random_hit += 1
        random_hit /= num_random
        return [precision, recall, f1, random_hit]


"""
add Hit@k
"""


def cal_hits_at_k(gold_answer_list: List[str], pred_answer_list: List[str], k=1) -> float:
    """
    Calculate Hits@k metric for evaluating the quality of predicted answers.

    :param gold_answer_list: The list of gold (true) answers.
    :param pred_answer_list: The list of predicted answers.
    :param k: The number of top-ranked items to consider for matching.
    :return: The Hits@k score.
    """
    # Ensure the predicted answers list has at least 'k' elements
    pred_answer_list = pred_answer_list[:k]

    # Check if any of the top 'k' predicted answers is in the gold answers list
    for pred_answer in pred_answer_list:
        if pred_answer in gold_answer_list:
            return 1.0

    # No matches found within top 'k' predictions
    return 0.0


def cal_metrics(golden_answers: List[List[str]], predictions: List[List[str]]):
    results = [CalculatePRF1(g, p) for g, p in zip(golden_answers, predictions)]
    if len(results) == 0:
        return [0.0] * 9
    ave_pre = sum([r[0] for r in results]) / len(results)
    ave_rec = sum([r[1] for r in results]) / len(results)

    # Macro F1 version 1: average F1 of each question
    average_f1 = sum([r[2] for r in results]) / len(results)

    # Macro F1 version 2: F1 of average precision and average recall
    f1_average = 2 * ave_pre * ave_rec / (ave_pre + ave_rec)

    # accuracy (ratio of questions answered exactly correctly)
    acc = sum([r[2] == 1.0 for r in results]) / len(results)

    # random Hits@1
    average_random_hit = sum([r[3] for r in results]) / len(results)

    # cal Hit@1 5 10
    hits = ["", "", ""]
    for idx, k in enumerate([1, 5, 10]):
        hits[idx] = sum([cal_hits_at_k(g, p, k) for g, p in zip(golden_answers, predictions)]) / len(results)

    # precision, recall, average_f1, f1_average, accuracy, hit1, hit5, hit10, random_hit
    return [
        ave_pre,
        ave_rec,
        average_f1,
        f1_average,
        acc,
        hits[0],
        hits[1],
        hits[2],
        average_random_hit,
    ]


if __name__ == "__main__":
    """
    python evaluation/webqsp.py
    """
    golden_answers = [
        ["a1", "a2", "a3", "a4"],
        ["a1", "a2", "a7"],
    ]
    predictions = [
        ["a1", "a2", "a3", "a4", "b1"],
        ["a1", "a2", "a5"],
    ]
    metrics = cal_metrics(golden_answers, predictions)
    print(metrics)
