import json
import re
from glob import glob

from evaluation.webqsp import cal_metrics


def cal_fb(dataset="webqsp"):
    if dataset == "webqsp":
        dirname = "save-qa-infer-dialog/webqsp/gpt-4-1106-preview"
    else:
        dirname = "save-qa-infer-dialog/cwq-addqtype/gpt-4-1106-preview"

    paths = glob(dirname + "/*.json")
    golden_answers = []
    predictions = []
    occur_in_question = []
    for p in paths:
        d = json.load(open(p))
        golden_entity = re.compile(r'type.object.name "(.*?)"@en').findall(d["sparql"])
        golden_entity = list(set(golden_entity))

        if golden_entity:
            matched = sum([1 for q in golden_entity if q.lower() in d["question"].lower()]) / len(
                golden_entity
            )
        else:
            matched = 0
        occur_in_question.append(matched)

        pred_entity = []
        for dia in d["dialog"]:
            if dia["role"] == "assistant":
                pred_entity += re.compile(r'type.object.name "(.*?)"@en').findall(dia["content"])
                if pred_entity:
                    break

        pred_entity = list(set(pred_entity))
        # pred_entity = [p for p in pred_entity if p and p in golden_entity]

        golden_answers.append(golden_entity)
        predictions.append(pred_entity)

    (
        precision,
        recall,
        average_f1,
        f1_average,
        accuracy,
        hit1,
        hit5,
        hit10,
        random_hit,
    ) = cal_metrics(golden_answers=golden_answers, predictions=predictions)
    ave_occur = sum(occur_in_question) / len(occur_in_question)
    line = f"precision: {precision}, recall: {recall}, average_f1: {average_f1}, f1_average: {f1_average}, accuracy: {accuracy}, hit1: {hit1}, hit5: {hit5}, hit10: {hit10}, random_hit: {random_hit}, ave_occur: {ave_occur}"
    print(dirname)
    print(line)
    print()


def cal_kqa():
    dirname = "save-qa-infer-dialog/kqapro-addqtype/gpt-4-1106-preview"
    paths = glob(dirname + "/*.json")
    golden_answers = []
    predictions = []

    # e.g. Q: 'What is the connection between Steve Jordan (the one whose position is tight end) to Phoenix (the one that is the twinned administrative body of Chengdu)?'
    # sparql has 2 entities: "Steve Jordan" and "Phoenix", than occur += 0.5
    occur_in_question = []
    for p in paths:
        d = json.load(open(p))
        # <pred:name> "Chengdu"
        golden_entity = re.compile(r'<pred:name> "(.*?)"').findall(d["sparql"])
        golden_entity = list(set(golden_entity))

        if golden_entity:
            matched = sum([1 for q in golden_entity if q.lower() in d["question"].lower()]) / len(
                golden_entity
            )
        else:
            matched = 0
        occur_in_question.append(matched)

        # 抽取所有dialog中 'role': 'assistant' 中的entity
        pred_entity = []
        for dia in d["dialog"]:
            if dia["role"] == "assistant":
                pred_entity += re.compile(r'<pred:name> "(.*?)"').findall(dia["content"])
                # if pred_entity:break

        # 可能有多余的，删掉？
        pred_entity = list(set(pred_entity))
        # pred_entity = [p for p in pred_entity if p and p in golden_entity]

        golden_answers.append(golden_entity)
        predictions.append(pred_entity)

    (
        precision,
        recall,
        average_f1,
        f1_average,
        accuracy,
        hit1,
        hit5,
        hit10,
        random_hit,
    ) = cal_metrics(golden_answers=golden_answers, predictions=predictions)
    ave_occur = sum(occur_in_question) / len(occur_in_question)
    line = f"precision: {precision}, recall: {recall}, average_f1: {average_f1}, f1_average: {f1_average}, accuracy: {accuracy}, hit1: {hit1}, hit5: {hit5}, hit10: {hit10}, random_hit: {random_hit}, ave_occur: {ave_occur}"
    print(dirname)
    print(line)
    print()


def cal_metaqa():
    test_data = {}
    for p in [
        "dataset_processed/metaqa/test/1-hop.json",
        "dataset_processed/metaqa/test/2-hop.json",
        "dataset_processed/metaqa/test/3-hop.json",
    ]:
        data = json.load(open(p))
        for d in data:
            test_data[d["id"]] = d
    dirname = "save-qa-infer-dialog/metaqa/gpt-4-1106-preview"
    paths = glob(dirname + "/*.json")
    golden_answers = []
    predictions = []
    occur_in_question = []
    for p in paths:
        d = json.load(open(p))
        raw_q = test_data[d["id"]]["raw_q"]
        # '[Caprice] is a film written by this person' -> 'Caprice'
        golden_entity = re.compile(r"\[(.*?)\]").findall(raw_q)
        golden_entity = list(set(golden_entity))

        if golden_entity:
            matched = sum([1 for q in golden_entity if q.lower() in d["question"].lower()]) / len(
                golden_entity
            )
        else:
            matched = 0
        occur_in_question.append(matched)

        pred_entity = []
        for dia in d["dialog"]:
            if dia["role"] == "assistant":
                pred_entity += re.compile(r'<name> "(.*?)"').findall(dia["content"])
                if pred_entity:
                    break

        pred_entity = list(set(pred_entity))
        # pred_entity = [p for p in pred_entity if p and p in golden_entity]

        golden_answers.append(golden_entity)
        predictions.append(pred_entity)

    (
        precision,
        recall,
        average_f1,
        f1_average,
        accuracy,
        hit1,
        hit5,
        hit10,
        random_hit,
    ) = cal_metrics(golden_answers=golden_answers, predictions=predictions)
    ave_occur = sum(occur_in_question) / len(occur_in_question)
    line = f"precision: {precision}, recall: {recall}, average_f1: {average_f1}, f1_average: {f1_average}, accuracy: {accuracy}, hit1: {hit1}, hit5: {hit5}, hit10: {hit10}, random_hit: {random_hit}, ave_occur: {ave_occur}"
    print(dirname)
    print(line)
    print()


if __name__ == "__main__":
    # python evaluation/entity_detection.py
    cal_fb("webqsp")
    cal_fb("cwq")
    cal_kqa()
    cal_metaqa()
