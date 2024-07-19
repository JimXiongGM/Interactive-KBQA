import os
import re
from glob import glob

import fire

from common.common_utils import colorful, read_json, save_to_json
from common.constant import TOOL_DESC_FULL_FB
from tool.action_execution import parse_action
from tool.openai import chatgpt


def _print_colorful(msg: dict):
    if msg["role"] == "user":
        print(colorful("Observation: ", color="yellow"))
    elif msg["role"] == "system":
        print(colorful("system: ", color="grey"))
    elif msg["role"] == "assistant":
        print(colorful("LLM: ", color="green"))
    else:
        raise ValueError(f"Unknown role: {msg['role']}")
    if "Follow the demos' format strictly!" not in msg["content"]:
        print(msg["content"])


def load_demos(qtype):
    demo_ids = []
    demos = []
    # fewshot_demo/cwq/dialog/comparative-01.txt
    for idx in ["01", "02"]:
        p = f"fewshot_demo/cwq/dialog/{qtype}-{idx}.txt"
        demo = open(p).readlines()
        _id = [line for line in demo if line.startswith("# ID:")]
        if _id:
            _id = _id[0].split("# ID:")[-1].strip()
            demo_ids.append(_id)
        demo = [i for i in demo if not i.startswith("#") and not i.startswith("SPARQL:")]
        demo = "".join(demo).strip()
        demos.append(demo)
    demos = "\n\n".join(demos)
    return demo_ids, demos


def extract_predicates(sparql_query):
    # \b represents a word boundary, ensuring that the matched text is the entire predicate name.
    predicate_pattern = re.compile(r"\bns:([\w.-]+)")
    predicates = re.findall(predicate_pattern, sparql_query)
    return set(predicates) - set(["type.object.name", "common.topic.notable_types"])


UNDERLINE_START = "\033[4m"
UNDERLINE_END = "\033[0m"


def add_underline(text):
    return UNDERLINE_START + text + UNDERLINE_END


def underline_predicates(text, predicates):
    for predicate in predicates:
        text = text.replace(predicate, add_underline(predicate))
    return text


def chat_anno(d, db, tooldesc_demos, model_name, save_dir, max_round_num=10):
    """
    curr_idx: used to index the rejected response
    """
    tooldesc_demos = tooldesc_demos.strip()
    question = d["question"]
    _id = d["id"]
    save_name = os.path.join(save_dir, f"{_id}.json")
    print("save_name: ", save_name)

    d["done"] = False
    if os.path.exists(save_name):
        d = read_json(save_name)
        if d["done"]:
            return
        messages = d["dialog"]
        [_print_colorful(msg) for msg in messages]
        print()
        completion_tokens = d["completion_tokens"][: len(messages)]
        prompt_tokens = d["prompt_tokens"][: len(messages)]
        rejects = [r for r in d["rejects"] if r[0] < len(messages)]
    else:
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": ""},
        ]
        completion_tokens = []
        prompt_tokens = []
        rejects = []

    if d["done"]:
        return

    # update tooldesc and demos
    messages[1]["content"] = tooldesc_demos + f"\n\nQ: {question}\nThought:"
    print(colorful("Q: ", color="green") + d["question"])
    print(colorful("Golden SPARQL:", color="green"))
    print(d["sparql"])
    print("ExecuteSPARQL('" + d["sparql"].replace("\n", " ")[41:] + "')")
    print(colorful("Golden answer:", color="green"))
    print(d["answers"])
    print()

    # for easy annotation
    predicates = extract_predicates(d["sparql"])

    # human add
    if messages[-1]["role"] == "assistant":
        observation = parse_action(messages[-1]["content"], db=db, execute=True)
        observation = str(observation).strip()
        messages.append({"role": "user", "content": "Observation: " + observation})
        print(colorful("Observation: ", color="yellow"))
        print(underline_predicates(observation, predicates))

    while len(messages) < max_round_num and not d["done"]:
        print(f"current len(messages)/max_round_num: {len(messages)}/{max_round_num}")
        response = chatgpt(
            model=model_name,
            messages=messages,
            stop=["\nObservation", "\nThought"],
            temperature=0,
            max_tokens=384,
            n=1,
        )

        completion_tokens.append(response["usage"]["completion_tokens"])
        prompt_tokens.append(response["usage"]["prompt_tokens"])

        out_thought_action = response["choices"][0]["message"]["content"].strip()
        out_thought_action = out_thought_action.replace("\n\n", "\n").split("\nObservation")[0]

        # add model info
        d["model_name"] = model_name
        d["completion_tokens"] = completion_tokens
        d["prompt_tokens"] = prompt_tokens

        # possible result
        observation_maybe = parse_action(out_thought_action, db=db, execute=True)
        observation_maybe = str(observation_maybe).strip()

        print(colorful("Possible ChatGPT:", color="yellow"))
        print(underline_predicates(out_thought_action.replace("\n", "\\n"), predicates))
        print(colorful("Possible observation:", color="yellow"))
        print(underline_predicates(observation_maybe, predicates))

        inp = input(colorful("Accept? ", color="red"))
        inp = inp.strip()
        if inp.lower() in ["1", "y", ""]:
            print("Accept")
            accept = out_thought_action
            Observation = observation_maybe
        elif inp.lower()[:4] in ["skip"]:
            d["skip_reason"] = inp[4:]
            d["done"] = True
            save_to_json(d, save_name, _print=False)
            return
        else:
            print(colorful("Reject", color="red"))
            accept = inp
            reject = out_thought_action
            rejects.append([len(messages), reject])
            Observation = parse_action(accept, db=db, execute=True)
            Observation = str(Observation).strip()
            print(colorful("Accepted observation:", color="blue"))
            print(underline_predicates(Observation, predicates))

        if not accept.startswith("Thought:"):
            accept = "Thought: " + accept
        messages.append({"role": "assistant", "content": accept})

        if Observation != "Done":
            messages.append({"role": "user", "content": "Observation: " + Observation})
        else:
            messages.append({"role": "user", "content": "Stop condition detected."})
            d["done"] = True

        d["dialog"] = messages
        d["rejects"] = rejects

        save_to_json(d, save_name, _print=False)
    inp = input("Continue? ")
    if inp.strip().lower() in ["n", "no"]:
        exit()


def run(qtype):
    """
    every type 50 cases.
    save to human-anno/cwq/[qtype]/{id}.json
    """
    assert qtype in [
        "comparative",
        "composition",
        "conjunction",
        "superlative",
    ]

    demo_ids, demos = load_demos(qtype)
    tooldesc_demos = TOOL_DESC_FULL_FB + "\n\n" + demos

    data = read_json(f"dataset_processed/cwq/train/{qtype}.json")

    skip_ids = []
    bad_num = 0
    paths = glob(f"human-anno/cwq/{qtype}/*.json")
    for p in paths:
        d = read_json(p)
        if d["done"]:
            skip_ids.append(d["id"])
        if "skip_reason" in d:
            bad_num += 1
    skip_ids = set(skip_ids)
    data = [d for d in data if d["id"] not in skip_ids and d["id"] not in demo_ids]
    remain_num = 50 - len(skip_ids) + bad_num

    print(f"len(data): {len(data)}")
    for d in data:
        if remain_num == 0:
            break
        print("-" * 50, d["id"], "-" * 50)
        print("Current:", qtype, "Len:", len(glob(f"human-anno/cwq/{qtype}/*.json")))
        print("Remain:", remain_num)
        chat_anno(
            d,
            db="fb",
            tooldesc_demos=tooldesc_demos,
            model_name="gpt-4-1106-preview",
            save_dir=f"human-anno/cwq/{qtype}",
            max_round_num=20,
        )
        remain_num -= 1


def check_data():
    """
    print
    - each qtype's valid num, skip num, average turn num
    check
    - dialog role: system, [user, assistant, user, assistant, ...]
    - last of dialog: content="Stop condition detected."
    - each content of assistant: "Thought: ", user: "Observation: "
    """
    qtypes = [
        "comparative",
        "composition",
        "conjunction",
        "superlative",
    ]
    for qtype in qtypes:
        paths = glob(f"human-anno/cwq/{qtype}/*.json")
        valid_num = 0
        skip_num = 0
        turn_num = 0
        for p in paths:
            d = read_json(p)
            if "skip_reason" in d:
                skip_num += 1
            else:
                valid_num += 1
                turn_num += len(d["dialog"])
                assert d["dialog"][0]["role"] == "system"
                assert d["dialog"][-1]["content"] == "Stop condition detected."
                for idx, turn in enumerate(d["dialog"][1:]):
                    if idx % 2 == 0:
                        assert turn["role"] == "user"
                        assert (
                            turn["content"].startswith("Observation: ")
                            or "Stop" in turn["content"]
                            or "Follow the demos' format strictly!" in turn["content"]
                        )
                    else:
                        assert turn["role"] == "assistant"
                        assert turn["content"].startswith("Thought: ")

        print(f"qtype: {qtype}")
        print(f"valid_num: {valid_num}")
        print(f"skip_num: {skip_num}")
        print(f"average turn num: {turn_num/valid_num}")
        print()


if __name__ == "__main__":
    """
    python llm_anno_human_cwq.py --qtype composition
    python llm_anno_human_cwq.py --qtype conjunction
    python llm_anno_human_cwq.py --qtype comparative
    python llm_anno_human_cwq.py --qtype superlative
    """
    fire.Fire(run)
    # check_data()
