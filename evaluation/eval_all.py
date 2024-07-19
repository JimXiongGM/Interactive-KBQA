import string
from collections import defaultdict
from copy import deepcopy
from glob import glob
from typing import List

import regex

from common.common_utils import colorful, read_json
from evaluation.webqsp import CalculatePRF1, cal_metrics


def clean_obs(result: str):
    """
    imi', 'Pasenge..." -> find last ', ' and remove the last.
    """
    if result.endswith("..."):
        idx = result.rfind("', '")
        result = result[:idx] + "']"
    return result


def clean_xsd(item):
    """
    ['"2004"^^xsd:gYear'] -> ['2004']
    """
    if isinstance(item, list):
        item = sorted(set([str(clean_xsd(i)) for i in item]))
    elif isinstance(item, str) and "^^xsd:" in item:
        item = item.split("^^xsd:")[0]
        item = item.strip('"')
    elif isinstance(item, bool):
        item = str(item)
    elif item is None:
        item = "_None"
    return item


def extract_dialog_last_observation(d):
    if not d:
        return None

    if isinstance(d["dialog"][-1], str):
        last_content = d["dialog"][-1]
    elif isinstance(d["dialog"][-1], dict):
        last_content = d["dialog"][-1]["content"]
    if last_content == "Stop condition detected.":
        # use the last Observation. 
        try:
            last_obs = d["dialog"][-3]["content"].replace("Observation: ", "").replace("\nThought:", "")
            last_obs = clean_obs(last_obs)
            pred_ans = eval(last_obs)
            pred_ans = pred_ans if isinstance(pred_ans, list) else [str(pred_ans)]
        except:
            # print("eval error: ", last_obs)
            pred_ans = ["_None"]
        return sorted(set(pred_ans), key=pred_ans.index) if isinstance(pred_ans, list) else [pred_ans]
    return ["_None"]


def _is_digit(s):
    try:
        s = float(s)
        return True
    except ValueError:
        return False


def _clean_number_and_str(pred_ans: List[str]):
    """
    rm numeric item if str item exists
    e.g.
        ["2", "usa"] -> ["usa"]
    """
    if len(pred_ans) > 1:
        if any([_is_digit(i) for i in pred_ans]):
            if any([not _is_digit(i) for i in pred_ans]):
                pred_ans = [i for i in pred_ans if not _is_digit(i)]
    return pred_ans


def extract_dialog_last_valid_observation(d):
    if not d:
        return None
    pred_ans = []
    for item in d["dialog"][::-1]:
        if (
            item["role"] == "user"
            and item["content"].startswith("Observation: ")
            and " | " not in item["content"]
        ):
            obs = item["content"].replace("Observation: ", "")
            if "?e" in obs:
                continue
            obs = clean_obs(obs)
            try:
                obs = eval(obs)
            except SyntaxError:
                continue

            if isinstance(obs, list) and len(obs) > 0 and not isinstance(obs[0], tuple):
                pred_ans = [clean_xsd(i) for i in obs]
                break
            elif isinstance(obs, bool):
                pred_ans = [obs]
                break
    pred_ans = sorted(set(pred_ans)) if len(pred_ans) > 0 else ["_None"]
    pred_ans = _clean_number_and_str(pred_ans)
    return pred_ans


def extract_dialog_last_valid_observation_kqa(d):
    if not d:
        return None
    pred_ans = []
    for item in d["dialog"][::-1]:
        if (
            item["role"] == "user"
            and item["content"].startswith("Observation: ")
            and " | " not in item["content"]
        ):
            obs = item["content"].replace("Observation: ", "")
            if "?e " in obs:
                continue
            obs = clean_obs(obs)
            try:
                obs = eval(obs)
            except SyntaxError:
                continue

            # "Observation: [('Bernalillo County', '3027.0')]" -> ['Bernalillo County', '3027.0']
            if isinstance(obs, list) and len(obs) == 1 and isinstance(obs[0], tuple):
                pred_ans = list(obs[0])
                break

            # for kqa
            if "pred:fact_r" in str(obs) and len(obs) > 3:
                # rm pred:fact*; replace statement_is_subject_of to zzz;
                obs = [tup for tup in obs if "pred:fact" not in tup[0]]
                for i in range(len(obs)):
                    if obs[i][0] == "statement_is_subject_of":
                        obs[i] = ("zzz_statement_is_subject_of", obs[i][1])
                # sort by the first element
                obs = sorted(obs, key=lambda x: x[0])[0]
                pred_ans = [obs[1]]
                break
            elif isinstance(obs, list) and len(obs) > 0 and not isinstance(obs[0], tuple):
                pred_ans = [clean_xsd(i) for i in obs]
                break
            elif isinstance(obs, bool):
                pred_ans = [obs]
                break
    pred_ans = sorted(set(pred_ans)) if len(pred_ans) > 0 else ["_None"]
    pred_ans = _clean_number_and_str(pred_ans)
    return pred_ans


def cal_dialog_info(data):
    """
    return:
        ave_turn: average all dialog turns
        success_rate: end with "Stop condition detected."
        ave_turn_done: average dialog turns for success dialog
    """
    # exclude the first "You are an AI assistant."
    ave_turn = sum([len(d["dialog"]) - 1 for d in data]) / len(data)
    success_rate = sum([d["dialog"][-1]["content"] == "Stop condition detected." for d in data]) / len(data)
    num_success = sum([d["dialog"][-1]["content"] == "Stop condition detected." for d in data])
    ave_turn_done = (
        sum([len(d["dialog"]) - 1 for d in data if d["dialog"][-1]["content"] == "Stop condition detected."])
        / num_success
    )
    return ave_turn, success_rate, ave_turn_done


def evaluation_webqsp(model_name, dirname="save-qa-infer-dialog", golden=False):
    """
    metrics: precision, recall, average_f1, f1_average, accuracy, hit1, hit5, hit10
    """
    save_dir = f"{dirname}/webqsp"
    if golden:
        save_dir += "-golden"
    save_dir += f"/{model_name}"

    print(f"save_dir: {save_dir}")
    paths = glob(save_dir + "/*.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        d = read_json(p)
        d["prediction"] = (
            extract_dialog_last_valid_observation(d) if "prediction" not in d else d["prediction"]
        )

        data.append(d)

    # [("type", data), ...]
    _data1 = [d for d in data if len(d["infer_chain"]) == 1]
    _data2 = [d for d in data if len(d["infer_chain"]) == 2]
    tp_data = [
        ("chain_len_1", _data1),
        ("chain_len_2", _data2),
        ("total", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        print(f"{tp}: {len(data)}", end="\t")
    print()

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answers"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        # cal metrics for webqsp and cwq; cal acc for kqapro and metaqa
        (
            precision,
            recall,
            average_f1,
            f1_average,
            accuracy,
            hit1,
            hit5,
            hit10,
            average_random_hit,
        ) = cal_metrics(golden_answers, predictions)
        if idx == 0:
            head = "type\tprecision\trecall\tave_f1\tf1_ave\taccuracy\thit@1\thit@5\thit@10\tRandomHit"
            if data and "dialog" in data[0]:
                head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
            print(colorful(head, "red"))

        line = f"{tp}\t{precision}\t{recall}\t{average_f1}\t{f1_average}\t{accuracy}\t{hit1}\t{hit5}\t{hit10}\t{average_random_hit}"

        # turn info
        if data and "dialog" in data[0]:
            ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
            line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"

        print(line)
    print()
    return line


# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    """
    copy from DeCAF:
    """
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def cal_exact_match_score(golden_answers: List[str], predictions: List[str]):
    if len(golden_answers) == 0:
        if len(predictions) == 0:
            return 1
        return 0
    for prediction in predictions:
        assert isinstance(prediction, str)
        em_score = ems(prediction, golden_answers)
        if em_score:
            return em_score
    return 0


def evaluation_cwq(
    model_name, dirname="save-qa-infer-dialog", addqtype=False, golden=False, setting="", valid_ids=None
):
    """
    metrics: exact_match
    setting: for ablation study
    """
    save_dir = f"{dirname}/cwq"
    if setting:
        save_dir += f"-{setting}"
    if addqtype:
        save_dir += "-addqtype"
    if golden:
        save_dir += "-golden"
    save_dir += f"/{model_name}"

    print(f"save_dir: {save_dir}")
    paths = glob(save_dir + "/*.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        qid = p.split("/")[-1].split(".json")[0]
        if valid_ids is not None and qid not in valid_ids:
            continue
        d = read_json(p)
        d["prediction"] = (
            extract_dialog_last_valid_observation(d) if "prediction" not in d else d["prediction"]
        )
        data.append(d)

    # [("type", data), ...]
    _data1 = [d for d in data if d["compositionality_type"] == "conjunction"]
    _data2 = [d for d in data if d["compositionality_type"] == "composition"]
    _data3 = [d for d in data if d["compositionality_type"] == "comparative"]
    _data4 = [d for d in data if d["compositionality_type"] == "superlative"]
    tp_data = [
        ("conjunction", _data1),
        ("composition", _data2),
        ("comparative", _data3),
        ("superlative", _data4),
        ("total", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        print(f"{tp}: {len(data)}", end="\t")
    print()

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answers"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        (
            precision,
            recall,
            average_f1,
            f1_average,
            accuracy,
            hit1,
            hit5,
            hit10,
            average_random_hit,
        ) = cal_metrics(golden_answers, predictions)

        # metrics: exact match, As long as any one of the golden answers is in the predicted answers,
        # it will be considered correct.
        # cwq paper code adds all aliases to golden answers to calculate "exact match"
        golden_answers_add_alias = []
        for d in data:
            aliases = d["answers"] if isinstance(d["answers"], list) else [d["answers"]]
            for a in d["answers_raw"]:
                aliases.append(a["answer"])
                aliases.extend(a["aliases"])
            aliases = sorted(set(aliases))
            golden_answers_add_alias.append(aliases)

        assert len(golden_answers_add_alias) == len(predictions)

        exact_match = (
            sum([cal_exact_match_score(gs, ps) for gs, ps in zip(golden_answers_add_alias, predictions)])
            / len(predictions)
            if len(predictions) > 0
            else 0
        )

        # print q types as header, acc as value
        if idx == 0:
            head = "type\tprecision\trecall\tave_f1\tf1_ave\taccuracy\thit@1\thit@5\thit@10\texact_match"
            if data and "dialog" in data[0]:
                head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
            print(colorful(head, "red"))

        line = f"{tp}\t{precision}\t{recall}\t{average_f1}\t{f1_average}\t{accuracy}\t{hit1}\t{hit5}\t{hit10}\t{exact_match}"

        # turn info
        if data and "dialog" in data[0]:
            ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
            line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"

        print(line)
    print()
    return line


# --- kqa
from datetime import date


def whether_equal(answer: str, pred: str):
    """
    check whether the two arguments are equal as attribute value
    """
    if isinstance(answer, list):
        answer = answer[0] if answer else "_None"
    if isinstance(pred, list):
        pred = pred[0] if pred else "_None"

    pred = pred.replace("_", " ")
    answer = answer.replace("_", " ")

    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = "{} {}".format(str(v), " ".join(u))
        except:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split("-")
            y_split = y.split("-")
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except:
            return False

    answer = truncate_float(answer)
    pred = truncate_float(pred)
    if equal_as_date(answer, pred):
        return True
    else:
        return answer == pred


def evaluation_kqapro(model_name, dirname="save-qa-infer-dialog", addqtype=False, setting="", valid_ids=None):
    """
    metrics: acc.
    must have keys:
        - answers: List[str]
        - prediction: List[str]
        - qtype
    """
    if setting:
        save_dir = f"{dirname}/kqapro-{setting}/{model_name}"
    elif addqtype:
        save_dir = f"{dirname}/kqapro-addqtype/{model_name}"
    else:
        save_dir = f"{dirname}/kqapro/{model_name}"

    print(f"save_dir: {save_dir}")
    paths = glob(save_dir + "/*.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        # debug
        if "val-208" in p:
            x = 1
        qid = p.split("/")[-1].split(".json")[0]
        if valid_ids is not None and qid not in valid_ids:
            continue
        d = read_json(p)
        if "prediction" not in d:
            pred = extract_dialog_last_valid_observation_kqa(d)
            d["prediction"] = pred
        data.append(d)

    # [("type", data), ...]
    # Count   QueryAttr   QueryAttrQualifier  QueryName   QueryRelation   QueryRelationQualifier  SelectAmong     SelectBetween    Verify
    for d in data:
        d["answers"] = d.pop("answer")
        d["answers"] = [d["answers"]] if isinstance(d["answers"], bool) else d["answers"]
    _data1 = [d for d in data if d["qtype"] == "Count"]
    _data2 = [d for d in data if d["qtype"] == "QueryAttr"]
    _data3 = [d for d in data if d["qtype"] == "QueryAttrQualifier"]
    _data4 = [d for d in data if d["qtype"] == "QueryName"]
    _data5 = [d for d in data if d["qtype"] == "QueryRelation"]
    _data6 = [d for d in data if d["qtype"] == "QueryRelationQualifier"]
    _data7 = [d for d in data if d["qtype"] == "SelectAmong"]
    _data8 = [d for d in data if d["qtype"] == "SelectBetween"]
    _data9 = [d for d in data if d["qtype"] == "Verify"]
    tp_data = [
        ("Count", _data1),
        ("QueryAttr", _data2),
        ("QueryAttrQualifier", _data3),
        ("QueryName", _data4),
        ("QueryRelation", _data5),
        ("QueryRelationQualifier", _data6),
        ("SelectAmong", _data7),
        ("SelectBetween", _data8),
        ("Verify", _data9),
        ("total", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        print(f"{tp}: {len(data)}", end="\t")
    print()

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answers"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        # follow the kqapro paper.
        # whether_equal for zip of golden_answers and predictions
        acc = (
            sum([whether_equal(goldn, pred) for goldn, pred in zip(golden_answers, predictions)])
            / len(golden_answers)
            if len(golden_answers) > 0
            else 0
        )

        # print q types as header, acc as value
        if idx == 0:
            head = "\t".join([i[0] for i in tp_data])
            if data and "dialog" in data[0]:
                head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
            print(colorful(head, "red"))
            line = [acc]
            continue
        line.append(acc)
        if idx == len(tp_data) - 1:
            line = "\t".join([str(i) for i in line])
            if data and "dialog" in data[0]:
                ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
                line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"
            print(line)

    print()
    return line


def evaluation_metaqa(model_name, dirname="save-qa-infer-dialog"):
    """
    metrics: precision, recall, average_f1, f1_average, accuracy, hit1, hit5, hit10
    """
    save_dir = f"{dirname}/metaqa/{model_name}"
    print(f"save_dir: {save_dir}")
    paths = glob(save_dir + "/*.json")
    data = []
    for p in paths:
        d = read_json(p)
        d["prediction"] = (
            extract_dialog_last_valid_observation(d) if "prediction" not in d else d["prediction"]
        )
        data.append(d)

    # [("type", data), ...]
    _data1 = [d for d in data if d["infer_chain"].count("_to_") == 1]
    _data2 = [d for d in data if d["infer_chain"].count("_to_") == 2]
    _data3 = [d for d in data if d["infer_chain"].count("_to_") == 3]

    tp_data = [
        ("1-hop", _data1),
        ("2-hop", _data2),
        ("3-hop", _data3),
        ("total", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        print(f"{tp}: {len(data)}", end="\t")
    print()

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answer"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        # cal metrics for metaqa and cwq; cal acc for kqapro and metaqa
        (
            precision,
            recall,
            average_f1,
            f1_average,
            accuracy,
            hit1,
            hit5,
            hit10,
            average_random_hit,
        ) = cal_metrics(golden_answers, predictions)
        if idx == 0:
            head = "type\tprecision\trecall\tave_f1\tf1_ave\taccuracy\thit@1\thit@5\thit@10\tRandomHit"
            # turn
            if data and "dialog" in data[0]:
                head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
            print(colorful(head, "red"))
        line = f"{tp}\t{precision}\t{recall}\t{average_f1}\t{f1_average}\t{accuracy}\t{hit1}\t{hit5}\t{hit10}\t{average_random_hit}"
        # turn
        if data and "dialog" in data[0]:
            ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
            line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"
        print(line)
    print()


# --- add
def cal_turns_avef1(dataset):
    print(f"dataset: {dataset}")
    if dataset in ["cwq", "kqapro"]:
        dataset = dataset + "-addqtype"

    p = f"save-qa-infer-dialog/{dataset}/gpt-4-1106-preview/*.json"
    paths = glob(p)

    turns_avef1 = defaultdict(list)
    for p in paths:
        d = read_json(p)
        d["prediction"] = (
            extract_dialog_last_valid_observation(d) if "prediction" not in d else d["prediction"]
        )

        if "kqapro" in dataset or "metaqa" in dataset:
            d["answers"] = d.pop("answer")
            d["answers"] = [d["answers"]] if isinstance(d["answers"], bool) else d["answers"]

        golden_answers = clean_xsd(d["answers"])
        predictions = clean_xsd(d["prediction"])
        precision, recall, f1, random_hit = CalculatePRF1(golden_answers, predictions)
        turns = len(d["dialog"])
        turns_avef1[turns].append(f1)

    # print by sorted turns, format is: turns num oneline, average f1 oneline
    for turns, f1s in sorted(turns_avef1.items()):
        print(turns, end="\t")
    print()
    for turns, f1s in sorted(turns_avef1.items()):
        print(sum(f1s) * 100 / len(f1s), end="\t")
    print()


if __name__ == "__main__":
    cal_turns_avef1("webqsp")
    cal_turns_avef1("cwq")
    cal_turns_avef1("kqapro")
    cal_turns_avef1("metaqa")
