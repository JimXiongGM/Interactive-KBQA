import os
import random
from collections import Counter
from glob import glob
from time import sleep
from typing import List

import fire
from loguru import logger

from common.common_utils import multi_process, read_json, save_to_json
from tool.openai_api import chatgpt

"""
gpt-4-1106-preview Answer (8 shots)
gpt-4-1106-preview SPARQL (8 shots)
gpt-4-1106-preview SPARQL (CoT, SC=6, 8 shots)
gpt-4-1106-preview Dialog (CoT, SC=6, 4 shots)

ToG:

Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.

My:
xxx The answer is: ['ans1', 'ans2']
"""


def load_test_data(dataset, case_num=None):
    assert dataset in [
        "webqsp",
        "cwq",
        "kqapro",
        "metaqa",
    ], f"dataset: {dataset} not supported."
    p = f"dataset_processed/{dataset}/test/*.json"
    paths = glob(p)
    data = []
    for p in paths:
        data += read_json(p)[:case_num]
    return data


def load_demo_data(dataset, setting) -> List[str]:
    p = f"fewshot_demo/{dataset}/{setting}/*.txt"
    demos = []
    for p in glob(p):
        lines = open(p).readlines()
        content = "".join([i for i in lines[1:] if not i.startswith("#")]).strip()
        demos.extend(content.split("\n\n"))
    return demos


def single(d, instruction, demos, model_name, n, save_dir):
    demos = demos.strip()
    q = d["question"].strip()
    prompt = f"{instruction}\n\n{demos}\n\nQ: {q}\nA: "

    response = chatgpt(
        prompt=prompt,
        model=model_name,
        temperature=0.7,
        top_p=1,
        n=n,
        stop=["\n\n", "\n"],
        max_tokens=256,
    )

    if response is None or "usage" not in response:
        print(f"response is None. id: {d['id']}")
        print(response)
        return

    d.pop("webqsp_question", None)
    d["completion_tokens"] = response["usage"]["completion_tokens"]
    d["prompt_tokens"] = response["usage"]["prompt_tokens"]
    d["response"] = [r["message"]["content"].strip() for r in response["choices"]]

    save_to_json(d, f"{save_dir}/{d['id']}.json", _print=False)


def run(dataset, setting, n=1, model_name="gpt-4-1106-preview"):
    """
    IO prompt: n=1 setting=io-answer
    CoT prompt: n=1 setting=cot-answer
    CoT + SC prompt: n=6 setting=cot-answer
    """
    assert dataset in [
        "webqsp",
        "cwq",
        "kqapro",
    ], "dataset must be one of webqsp, cwq, kqapro"
    assert setting in ["cot-answer", "io-answer"]

    if setting == "cot-answer":
        if dataset == "kqapro":
            instruction = "This is a question answering task. Given a Question, you need to write out the reasoning processes and the answers in python list format. Follow the demos' format strictly."
        else:
            instruction = "This is a question answering task. For some reasons, please assume you are in the year 2015 and unaware of what the future holds. Given a Question, you need to write out the reasoning processes and the answers in python list format. Follow the demos' format strictly."
    else:
        instruction = "This is a question answering task. Given a Question, you need to write out the answers in python list format. Follow the demos' format strictly."

    data = load_test_data(dataset)[:]
    demos = load_demo_data(dataset, setting)
    random.seed(42)
    random.shuffle(demos)
    demos = "\n\n".join(demos)
    print(demos)
    sleep(5)

    setting += f"-n{n}"
    save_dir = f"save-qa-infer-directly/{dataset}/{setting}/"
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
        process_function=single,
        cpu_num=30,
        debug=0,
        dummy=True,
        instruction=instruction,
        demos=demos,
        model_name=model_name,
        n=n,
        save_dir=save_dir,
    )


# ----------------- for cot answer -----------------
def parse_prediction_cot_answer(d):
    answers = [i.split("answer is:")[-1].strip() for i in d["response"] if "answer is:" in i]
    eval_answers = []
    for a in answers:
        try:
            a = eval(a)
            # exclude ellipsis
            a = [i for i in a if type(i) != type(Ellipsis)]
            if a and isinstance(a[0], list):
                a = a[0]
            eval_answers.append(tuple(a))
        except:
            try:
                a = a = (
                    a.split("']")[0]
                    .replace("'s", "\\'s")
                    .replace("'t", "\\'t")
                    .replace("' ", "\\' ")
                    .replace("O'", "O\\'")
                    .strip()
                    + "']"
                )
                a = eval(a)
            except:
                print("eval error. ID:", d["id"], "answer:", a)
    # count
    counter = Counter(eval_answers).most_common()
    d["prediction"] = sorted(counter[0][0]) if counter else ["_None"]
    return d


def post_process():
    paths = (
        glob("save-qa-infer-directly/webqsp/*-answer-*/gpt-4-1106-preview/*.json")
        + glob("save-qa-infer-directly/cwq/*-answer-*/gpt-4-1106-preview/*.json")
        + glob("save-qa-infer-directly/kqapro/*-answer-*/gpt-4-1106-preview/*.json")
    )
    for p in paths:
        d = read_json(p)
        d = parse_prediction_cot_answer(d)
        save_to_json(d, p, _print=False)


if __name__ == "__main__":
    """
    # webqsp:
    python llm_infer_directly.py --dataset webqsp --model_name gpt-4-1106-preview --setting io-answer
    python llm_infer_directly.py --dataset webqsp --model_name gpt-4-1106-preview --setting cot-answer --n 1
    python llm_infer_directly.py --dataset webqsp --model_name gpt-4-1106-preview --setting cot-answer --n 6

    # cwq:
    python llm_infer_directly.py --dataset cwq --model_name gpt-4-1106-preview --setting io-answer
    python llm_infer_directly.py --dataset cwq --model_name gpt-4-1106-preview --setting cot-answer --n 1
    python llm_infer_directly.py --dataset cwq --model_name gpt-4-1106-preview --setting cot-answer --n 6

    # kqapro
    python llm_infer_directly.py --dataset kqapro --setting io-answer
    python llm_infer_directly.py --dataset kqapro --setting cot-answer --n 1
    python llm_infer_directly.py --dataset kqapro --setting cot-answer --n 6
    """
    fire.Fire(run)
