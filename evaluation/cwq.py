import json
from glob import glob

import fire


def extract_LLM_answer(d):
    if not d:
        return None

    if isinstance(d["dialog"][-1], str):
        last_content = d["dialog"][-1]
        last_answer = d["dialog"][-2]
    elif isinstance(d["dialog"][-1], dict):
        last_content = d["dialog"][-1]["content"]
        last_answer = d["dialog"][-2]["content"]
    if last_content == "Stop condition detected.":
        # use the last Observation
        if "answer is" in last_answer and "Action: Done" in last_answer:
            try:
                last_obs = d["dialog"][-3]["content"].replace("Observation: ", "").replace("\nThought:", "")
                pred_ans = eval(last_obs)
                if isinstance(pred_ans, list):
                    new_pred_ans = []
                    for _var, ans in pred_ans:
                        # [('?x', 'm.07t7hy "Shinzō Abe"@en')]
                        white_idx = ans.find(" ")
                        ans = ans[white_idx + 1 :].replace("@en", "").strip('"')
                        new_pred_ans.append(ans)
                    pred_ans = new_pred_ans
            except:
                # print("eval error: ", last_obs)
                pred_ans = ["_None"]
            return sorted(set(pred_ans))
    return ["_None"]


def extract_golden_answers(d):
    ans = [a["answer"] for a in d["answers"]]
    return ans


def main(num=None):
    """
    python evaluation/kqapro.py --num=100
    """
    _paths = ["save/cwq/4types-gpt-3.5-turbo-16k-v1.0"]
    for _i in range(len(_paths)):
        paths = glob(_paths[_i] + "/train/*.jsonl")[:]
        accu_good, total = 0, 0
        for path in paths:
            print(path.split("cwq/")[-1], end="  ")
            data = [json.loads(i) for i in open(path).readlines()]
            data = [d for d in data if d]
            predictions = [extract_LLM_answer(d) if d else ["None"] for d in data[:num]]
            goldens = [extract_golden_answers(d) for d in data[:num]]
            assert len(predictions) == len(goldens)
            good = sum([1 if set(p) == set(g) else 0 for p, g in zip(predictions, goldens)])
            print("total:", len(goldens), end="  ")
            print("acc:", round(good / len(goldens), 4))
            accu_good += good
            total += len(goldens)
        print("total items:", total)
        print("total acc:", round(accu_good / total, 4))
        print()


if __name__ == "__main__":
    main()
    # fire.Fire(main)
