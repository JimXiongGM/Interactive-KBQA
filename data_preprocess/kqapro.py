import json
import random
from copy import deepcopy

from common.common_utils import multi_process, save_to_json
from tool.actions_kqapro import init_kqapro_actions


SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_kqapro_actions()


def _single(d):
    d["answer"] = ExecuteSPARQL(d["sparql"], str_mode=False)
    return d


def split_data_by_type(split="train"):
    """
    do:
        - run sparql to get answers, replace the given answer.
        - add qtype
        - train/val random sample 200 for each type
    """
    data = json.load(open(f"dataset/KQA-Pro-v1.0/{split}.json"))

    type_counter = {
        "QueryRelationQualifier": 1004,
        "Count": 1318,
        "QueryRelation": 1786,
        "SelectBetween": 1604,
        "QueryAttr": 1418,
        "QueryAttrQualifier": 1214,
        "SelectAmong": 548,
        "QueryName": 1458,
        "Verify": 1447,
    }

    random.seed(123)
    random.shuffle(data)
    _num = 9e9 if split == "val" else 200
    counter_remain = {k: _num for k in type_counter.keys()}
    data2 = []
    for d in data:
        function = d["program"][-1]["function"]
        function = "QueryName" if function == "What" else function
        function = (
            "Verify" if function in ["VerifyStr", "VerifyNum", "VerifyYear", "VerifyDate"] else function
        )
        if counter_remain[function] > 0:
            counter_remain[function] -= 1
            d = deepcopy(d)
            d.pop("program", None)
            d.pop("choices", None)
            d["qtype"] = function
            data2.append(d)

    data3 = multi_process(data2, _single, cpu_num=16, dummy=True)
    for _type in type_counter.keys():
        _data = [d for d in data3 if d["qtype"] == _type]
        split = split if split == "train" else "test"
        save_to_json(_data[:100], f"dataset_processed/kqapro/{split}/{_type}.json")


if __name__ == "__main__":
    """
    python data_preprocess/kqapro.py
    """
    split_data_by_type(split="val")
