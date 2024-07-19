import os
from glob import glob

import fire
from loguru import logger

from common.common_utils import multi_process, read_json
from llm_infer_directly import load_test_data
from tool.action_execution import chat_with_LLM


def run(dataset, model_name, debug=True, case_num=None):
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

    save_dir = f"save-qa-infer-dialog-finetuned/{dataset}/{model_name}"
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
        process_function=chat_with_LLM,
        cpu_num=5,
        debug=debug,
        dummy=True,
        # func params
        db=db,
        model_name=model_name,
        max_round_num=10,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    """
    python llm_infer_finetuned_dialog.py --dataset webqsp --model_name Merge-Llama-2-7b-hf-full-zero3-epoch10 --debug False
    python llm_infer_finetuned_dialog.py --dataset webqsp --model_name Merge-Mistral-7B-v0.1-full-zero3-epoch10 --debug False

    python llm_infer_finetuned_dialog.py --dataset cwq --model_name Merge-Llama-2-7b-hf-full-zero3-epoch10 --debug False
    python llm_infer_finetuned_dialog.py --dataset cwq --model_name Merge-Mistral-7B-v0.1-full-zero3-epoch10 --debug False

    python llm_infer_finetuned_dialog.py --dataset kqapro --model_name Merge-Llama-2-7b-hf-full-zero3-epoch10 --debug False
    python llm_infer_finetuned_dialog.py --dataset kqapro --model_name Merge-Mistral-7B-v0.1-full-zero3-epoch10 --debug False

    python llm_infer_finetuned_dialog.py --dataset metaqa --model_name Mistral-7B-v0.1-full-zero3-epoch10 --debug False
    """
    fire.Fire(run)
