import json
import os
from glob import glob

import fire
import requests
from loguru import logger
from tqdm import tqdm

from common.common_utils import colorful, multi_process, read_json, save_to_json
from evaluation.eval_all import evaluation_cwq, evaluation_kqapro, evaluation_metaqa, evaluation_webqsp
from llm_infer_directly import load_test_data

"""
For fine-tuned LLM
"""
headers = {"Content-Type": "application/json", "accept": "application/json"}


def infer_lf(d, db, save_dir):
    """
    curl -X POST "http://192.168.4.200:18100/fb" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"question": "what is the capital of china?", "stop": ["\n"]}'
    """
    question = d["question"].strip()
    url = f"http://192.168.4.200:18100/{db}"
    data = {"question": question, "stop": ["\n"], "db": db}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()

    d["response"] = response

    save_to_json(d, f"{save_dir}/{d['id']}.json")


def run(dataset, model_name, debug=True, case_num=None):
    """
    model_name: Merge-Mistral-7B-v0.1-full-zero3-epoch10
    case_num = 200  # None: unlimit"train"  # "train":xx, "val":xx
    """
    assert dataset in [
        "webqsp",
        "cwq",
        "kqapro",
        "metaqa",
    ], "dataset must be one of webqsp, cwq, kqapro, metaqa"
    if dataset in ["webqsp", "cwq"]:
        db = "fb"
    elif dataset == "kqapro":
        db = "kqapro"
    else:
        db = "metaqa"

    data = load_test_data(dataset, case_num=case_num)

    save_dir = f"save-qa-infer-lf-finetuned/{dataset}/{model_name}"
    logger.info(f"saving to: {save_dir}")

    skip_ids = []
    if os.path.exists(save_dir):
        paths = glob(save_dir + "/*.json")
        skip_ids += [read_json(p)["id"] for p in paths]

    skip_ids = set(skip_ids)
    logger.info(f"Skip id: {len(skip_ids)}")
    data = [d for d in data if d["id"] not in skip_ids]
    logger.info(f"remain len(data): {len(data)}")

    multi_process(
        items=data,
        process_function=infer_lf,
        cpu_num=8,
        debug=debug,
        dummy=True,
        # func params
        db=db,
        save_dir=save_dir,
    )


def cache_execution_fb():
    from tool.actions_fb import init_fb_actions

    SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_fb_actions()

    paths = glob("save-qa-infer-lf-finetuned/webqsp/**/*.json") + glob(
        "save-qa-infer-lf-finetuned/cwq/**/*.json"
    )
    for p in tqdm(paths):
        d = read_json(p)
        if "prediction" in d:
            continue
        d["prediction"] = ExecuteSPARQL(d["response"]["choices"][0]["message"]["content"], str_mode=False)
        save_to_json(d, p, _print=False)


def cache_execution_kqapro():
    from tool.actions_kqapro import init_kqapro_actions

    SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_kqapro_actions()

    paths = glob("save-qa-infer-lf-finetuned/kqapro/**/*.json")
    for p in tqdm(paths):
        d = read_json(p)
        # if "prediction" in d:
        #     continue
        pred = ExecuteSPARQL(d["response"]["choices"][0]["message"]["content"], str_mode=False)
        if not isinstance(pred, list):
            pred = [pred]
        d["prediction"] = pred
        save_to_json(d, p, _print=False)


if __name__ == "__main__":
    """
    python llm_infer_finetuned_lf.py --dataset webqsp --model_name Merge-Llama-2-7b-hf-full-zero3-epoch10 --debug False
    python llm_infer_finetuned_lf.py --dataset cwq --model_name Merge-Llama-2-7b-hf-full-zero3-epoch10 --debug False

    python llm_infer_finetuned_lf.py --dataset webqsp --model_name Merge-Mistral-7B-v0.1-full-zero3-epoch10 --debug False
    python llm_infer_finetuned_lf.py --dataset cwq --model_name Merge-Mistral-7B-v0.1-full-zero3-epoch10 --debug False
    """
    fire.Fire(run)

    # execute SPAQL
    # cache_execution_fb()
    # cache_execution_kqapro()

    # eval
    # python llm_infer_finetuned_lf.py

    # print(colorful("webqsp"))
    # evaluation_webqsp(dirname="save-qa-infer-lf-finetuned", model_name="Merge-Mistral-7B-v0.1-full-zero3-epoch10")

    # print(colorful("cwq"))
    # evaluation_cwq(dirname="save-qa-infer-lf-finetuned", model_name="Merge-Mistral-7B-v0.1-full-zero3-epoch10")

    # print(colorful("kqapro"))
    # evaluation_kqapro(dirname="save-qa-infer-lf-finetuned", model_name="Mistral-7B-v0.1-full-zero3-epoch10")
