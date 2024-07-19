from glob import glob

from common.common_utils import colorful, read_json

paths = glob("human-anno/webqsp/*/*.json")
for p in paths:
    print(p)
    d = read_json(p)
    for rej_idx, rej in d["rejects"]:
        d["dialog"][rej_idx]["reject"] = rej
    for round_id, dia in enumerate(d["dialog"]):
        role, content = dia["role"], dia["content"]
        if role == "assistant":
            role = colorful(role, color="green")
        else:
            role = colorful(role, color="yellow")
        print(f"No {round_id}. {role}: {content}")
    print("-" * 50)
    inp = input("Enter to continue, q to exit.")
    if inp.lower() == "q":
        break
