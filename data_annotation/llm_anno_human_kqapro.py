import os
import re
from glob import glob

import fire

from common.common_utils import colorful, read_json, save_to_json
from common.constant import TOOL_DESC_FULL_KQAPRO
from tool.action_execution import parse_action
from tool.openai_api import chatgpt


def load_demos(_type):
    demo_id = []
    demos = []
    # fewshot_demo/kqapro/dialog/Count-01.txt
    paths = [f"fewshot_demo/kqapro/dialog/{_type}-01.txt", f"fewshot_demo/kqapro/dialog/{_type}-02.txt"]
    for p in paths:
        with open(p) as f:
            demo = f.readlines()
        _id = [line for line in demo if line.startswith("# ID:")]
        _id = _id[0].split("# ID:")[-1].strip()
        demo_id.append(_id)
        demo = [i for i in demo if not i.startswith("#")]
        demo = "".join(demo).strip()
        demos.append(demo)
    demos = "\n\n".join(demos)
    return demo_id, demos


def clean_sparal(text):
    """
    replace multiple spaces to one space
    rename ?e_ to ?e
    """
    text = re.sub(r"\s+", " ", text)
    text = (
        text.replace("\n", " ")
        .replace("?e_", "?e")
        .replace("?c_", "?c")
        .replace("?pv_", "?pv")
        .replace("?v_", "?v")
    )
    return text


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


def extract_predicates(sparql_query):
    # match all predicates in <>
    predicate_pattern = re.compile(r"<([^<>]*)>")
    predicates = re.findall(predicate_pattern, sparql_query)
    predicates = [f"<{p}>" for p in predicates if not p.startswith("pred:")]
    return set(predicates)


# xxx = '''SELECT DISTINCT ?qpv WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> "city" . ?e <age_of_consent> ?pv1 . ?pv1 <pred:unit> "years old" . ?pv1 <pred:value> ?v1 . FILTER ( ?v1 != "17.1"^^xsd:double ) . ?e <Human_Development_Index> ?pv . ?pv <pred:unit> "1" . ?pv <pred:value> "0.91"^^xsd:double . [ <pred:fact_h> ?e ; <pred:fact_r> <Human_Development_Index> ; <pred:fact_t> ?pv ] <point_in_time> ?qpv . }'''
# r = extract_predicates(xxx)
# x=1

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
    print("ExecuteSPARQL('" + clean_sparal(d["sparql"]) + "')")
    print(colorful("Golden answer:", color="green"))
    print(d["answers"] if "answers" in d else d["answer"])
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
    save to human-anno/kqapro/[qtype]/{id}.json
    """
    assert qtype in [
        "Count",
        "QueryAttrQualifier",
        "QueryAttr",
        "QueryName",
        "QueryRelationQualifier",
        "QueryRelation",
        "SelectAmong",
        "SelectBetween",
        "Verify",
    ]

    demo_ids, demos = load_demos(qtype)
    tooldesc_demos = TOOL_DESC_FULL_KQAPRO + "\n\n" + demos

    print("tooldesc_demos")
    print(tooldesc_demos)

    data = read_json(f"dataset_processed/kqapro/train/{qtype}.json")

    skip_ids = []
    bad_num = 0
    paths = glob(f"human-anno/kqapro/{qtype}/*.json")
    for p in paths:
        d = read_json(p)
        if d["done"]:
            skip_ids.append(d["id"])
        if "skip_reason" in d:
            bad_num += 1
    data_tp = [d for d in data if d["id"] not in skip_ids + demo_ids]
    remain_num = 50 - len(skip_ids) + bad_num

    print(f"len(data_tp): {len(data_tp)}")
    for d in data_tp:
        if remain_num == 0:
            break

        print("-" * 50, d["id"], "-" * 50)
        print("Current:", qtype, "Len:", len(glob(f"human-anno/kqapro/{qtype}/*.json")))
        print("Remain:", remain_num)
        chat_anno(
            d,
            db="kqapro",
            tooldesc_demos=tooldesc_demos,
            model_name="gpt-4-1106-preview",
            save_dir=f"human-anno/kqapro/{qtype}",
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
        "Count",
        "QueryAttrQualifier",
        "QueryAttr",
        "QueryName",
        "QueryRelationQualifier",
        "QueryRelation",
        "SelectAmong",
        "SelectBetween",
        "Verify",
    ]
    for qtype in qtypes:
        paths = glob(f"human-anno/kqapro/{qtype}/*.json")
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
                for idx, turn in enumerate(d["dialog"][2:]):
                    if idx % 2 == 1:
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
    python llm_anno_human_kqapro.py --qtype Count
    python llm_anno_human_kqapro.py --qtype QueryAttrQualifier
    python llm_anno_human_kqapro.py --qtype QueryAttr
    python llm_anno_human_kqapro.py --qtype QueryName
    python llm_anno_human_kqapro.py --qtype QueryRelationQualifier
    python llm_anno_human_kqapro.py --qtype QueryRelation
    python llm_anno_human_kqapro.py --qtype SelectAmong
    python llm_anno_human_kqapro.py --qtype SelectBetween
    python llm_anno_human_kqapro.py --qtype Verify
    """
    fire.Fire(run)
    # run("Count")
    # check_data()
