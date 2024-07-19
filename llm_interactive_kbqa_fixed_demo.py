import os
import time
from glob import glob

import fire
from loguru import logger

from common.common_utils import multi_process, read_json
from common.constant import TOOL_DESC_FULL_FB, TOOL_DESC_FULL_KQAPRO
from evaluation.eval_all import evaluation_cwq, evaluation_kqapro
from llm_infer_directly import load_test_data
from tool.action_execution import chat_with_LLM

"""
In this document, we did not employ the question type classifier.
    - for CWQ, we directly utilized the fixed 4-shot demo, meaning that one demo was selected for each question type.
    - for kqa pro, due to the high cost of incorporating demos for all 9 question types into the prompt simultaneously, we also opted for the fixed 4-shot demo, corresponding to two settings.
"""


def load_fewshot_demo_dialog(dataset, setting):
    """
    cwq: fewshot_demo/cwq/dialog-fix-4-shot
    kqapro: fewshot_demo/kqapro/dialog-fix-4-shot-01; fewshot_demo/kqapro/dialog-fix-4-shot-02
    """
    dir_patt = f"fewshot_demo/{dataset}/dialog-fix-{setting}/*.txt"
    logger.warning(f"dir_patt: {dir_patt}")
    paths = glob(dir_patt)

    demos = []
    for p in paths:
        lines = open(p).readlines()
        lines = [i for i in lines if not i.startswith("#")]
        content = "".join(lines).strip()
        _demos = content.split("\n\n")
        demos.extend(_demos)

    assert len(demos) == 4, f"len(demos): {len(demos)}"
    return demos


def run(dataset, model_name, debug=True, case_num=10, setting=""):
    """
    cwq: fewshot_demo/cwq/dialog-fix-4-shot
    kqapro: fewshot_demo/kqapro/dialog-fix-4-shot-01; fewshot_demo/kqapro/dialog-fix-4-shot-02

    save to:
    cwq: save-qa-infer-dialog/cwq-4-shot/gpt-4-1106-preview
    kqapro: save-qa-infer-dialog/kqapro-4-shot-01; save-qa-infer-dialog/kqapro-4-shot-02
    """
    assert dataset in ["cwq", "kqapro"]
    if dataset in ["cwq"]:
        db = "fb"
        _desc = TOOL_DESC_FULL_FB
        assert setting in ["4-shot", "zero-shot"]
    elif dataset == "kqapro":
        db = "kqapro"
        _desc = TOOL_DESC_FULL_KQAPRO
        assert setting in ["4-shot-01", "4-shot-02", "zero-shot"]

    skip_ids = []
    if setting == "zero-shot":
        add_inst = "Follow the demos' format strictly: the format should be like this: Q:...\\nThought:...\\nAction:...\\nObservation:...\\nThought:..."
        add_inst += " You MUST provide ONLY ONE action in each turn like this: Action: SearchNodes(\" ... \"), or Action: SearchGraphPatterns('...'), or Action: ExecuteSPARQL(' ... '). If the observation is the answer, you should use Action: Done to stop. You must use Python syntax to call these tools, paying special attention to escaping the quotation marks."
        tooldesc_demos = _desc.replace("Follow the demos' format strictly!", add_inst)
    else:
        demos = load_fewshot_demo_dialog(dataset=dataset, setting=setting)
        tooldesc_demos = _desc + "\n\n" + "\n\n".join(demos)

    print("tooldesc_demos")
    print(tooldesc_demos)
    print()

    # NOTE: only use test set
    data = load_test_data(dataset, case_num=case_num)

    print(f"len(data): {len(data)}")

    _name = f"{dataset}-{setting}"

    save_dir = f"save-qa-infer-dialog/{_name}/" + model_name.replace("/", "-")
    logger.info(f"saving to: {save_dir}")

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
        cpu_num=10,
        debug=debug,
        dummy=True,
        # func params
        db=db,
        model_name=model_name,
        tooldesc_demos=tooldesc_demos,
        max_round_num=10,
        save_dir=save_dir,
    )


def extract_ids_from_path(path):
    path = path + "/*.json"
    path = glob(path)
    ids = [p.split("/")[-1].replace(".json", "") for p in path]
    return ids


if __name__ == "__main__":
    """
    # ------------------- cwq ------------------- #
    python llm_infer_baseline_dialog_ablation.py --debug False --dataset=cwq --model_name=gpt-4-1106-preview --setting=zero-shot --case_num=25
    python llm_infer_baseline_dialog_ablation.py --debug False --dataset=cwq --model_name=gpt-4-1106-preview --setting=4-shot --case_num=25

    # ------------------- kqapro ------------------- #
    python llm_infer_baseline_dialog_ablation.py --debug False --dataset=kqapro --model_name=gpt-4-1106-preview --setting=zero-shot --case_num=11
    python llm_infer_baseline_dialog_ablation.py --debug False --dataset=kqapro --model_name=gpt-4-1106-preview --setting=4-shot-01 --case_num=11
    python llm_infer_baseline_dialog_ablation.py --debug False --dataset=kqapro --model_name=gpt-4-1106-preview --setting=4-shot-02 --case_num=11
    """
    fire.Fire(run)
