import os
import random

from common.common_utils import multi_process, read_json, save_to_json
from data_preprocess.cwq import replace_mid
from tool.actions_fb import init_fb_actions

"""
Require:
    - unzip [WebQSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) to `dataset/WebQSP` folder.

Steps:
    1. replace mid
    2. run replaceed sparqls to get answers, replace the given answer.
    3. check the error answers due to mid to name mismatch.
    4. save to json.

Output:
    dataset_processed/webqsp/test/chain_len_1.json
    dataset_processed/webqsp/test/chain_len_2.json
"""

SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_fb_actions()

err_idx = 0


def _single(d):
    global err_idx
    RawQuestion = d["RawQuestion"]
    parse = d["Parses"][0]
    _id = parse["ParseId"]

    # debug
    # if _id != "WebQTrn-74.P0":
    #     return None

    raw_sparql = parse["Sparql"]
    if "#MANUAL" in raw_sparql:
        return None

    # keep infer chain
    infer_chain = parse["InferentialChain"]
    if len(infer_chain) not in [1, 2]:
        print(f"len(infer_chain): {len(infer_chain)}")
        return None

    raw_ans = ExecuteSPARQL(raw_sparql, str_mode=False)

    # replace mid to name in sparql
    sparql, var_x_name = replace_mid(raw_sparql, _return_var_x_name=True)
    answers = ExecuteSPARQL(sparql, str_mode=False)

    if not raw_ans or not answers:
        print(f"raw_ans: {raw_ans}")
        print(f"answers: {answers}")
        return None

    is_good = False
    if isinstance(answers, list):
        # check, all raw_ans should be in answers
        is_good = len(set(raw_ans) - set(answers + [var_x_name])) == 0

    # error
    if isinstance(answers, str) or not is_good:
        err_file = open("dataset_processed/webqsp/err_mid_to_name_mismatch.txt", "a", encoding="utf-8")
        err_idx += 1
        print(f"error No. {err_idx}: {_id}", file=err_file)
        print(f"answers type: {type(answers)}; is_good: {is_good}", file=err_file)
        if isinstance(answers, str):
            print(f"answers: {answers}", file=err_file)
            answers = [answers]
        if not is_good:
            print("answers in raw_ans but not in answers:", file=err_file)
            print(set(raw_ans) - set(answers + [var_x_name]), file=err_file)
        print(RawQuestion, file=err_file)
        print(file=err_file)
        print(parse["Sparql"], file=err_file)
        print(file=err_file)
        print(sparql, file=err_file)
        print(file=err_file)
        print("-*" * 50, file=err_file)
        err_file.close()
        return None
    answers = [a[1:-4] if a.endswith('"@en') else a for a in answers if a]
    answers = list(set(answers))
    tmp = {
        "id": _id,
        "question": RawQuestion,
        "sparql_raw": raw_sparql,
        "sparql": sparql,
        "answers": answers,
        "infer_chain": infer_chain,
    }
    return tmp


def split_data_by_type(split="test"):
    """
    do:
        - replace mid in sparql.
        - run sparql to get answers, replace the given answer.

    save to:
        dataset_processed/webqsp/test/chain_len_1.json
        dataset_processed/webqsp/test/chain_len_2.json

    len
        train: 3098 -> 3060
        test: 1639 -> 1620
    """
    os.makedirs("dataset_processed/webqsp", exist_ok=True)
    data = read_json(f"dataset/WebQSP/dataset_processed/WebQSP.{split}.json")["Questions"]
    random.seed(123)
    random.shuffle(data)
    res = multi_process(
        items=data[:500],
        process_function=_single,
        cpu_num=15,
        debug=0,
        dummy=True,
    )
    res = [i for i in res if i]

    chain_len_1s = []
    chain_len_2s = []
    for d in res:
        inferchain = d["infer_chain"]
        if len(inferchain) == 1 and len(chain_len_1s) < 150:
            chain_len_1s.append(d)
        elif len(inferchain) == 2 and len(chain_len_2s) < 150:
            chain_len_2s.append(d)
        else:
            pass

    save_to_json(chain_len_1s, f"dataset_processed/webqsp/{split}/chain_len_1.json")
    print(f"len(chain_len_1s): {len(chain_len_1s)}")
    save_to_json(chain_len_2s, f"dataset_processed/webqsp/{split}/chain_len_2.json")
    print(f"len(chain_len_2s): {len(chain_len_2s)}")


if __name__ == "__main__":
    """
    python data_preprocess/webqsp.py
    """
    split_data_by_type("test")
