import os
import random
import re

from tqdm import tqdm

from common.common_utils import multi_process, read_json, save_to_json
from tool.actions_fb import init_fb_actions
from tool.client_freebase import FreebaseClient

SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_fb_actions()


"""
clean the weird char in CWQ.
"""

fb = FreebaseClient("http://localhost:9501/sparql")
_pattern = re.compile(r"ns:([mg]\..*?) ")

re_no_chn_en_num = "[^(a-zA-Z0-9\u4e00-\u9fa5 )|^(’·°–!\"#$%&'()*+,-./:;<=>?@，。?★、…【】（）：《》？“”‘’！[\\]^_`{|}~)]"
re_no_chn_en_num = re.compile(re_no_chn_en_num)


def _argmin(values):
    min_idx = -1
    min_val = 9e9
    for i, v in enumerate(values):
        min_idx = i if v < min_val else min_idx
        min_val = v if v < min_val else min_val
    return min_idx


def find_special_char(text):
    """
    find all non-Chinese and non-English characters
    """
    res = re.findall(pattern=re_no_chn_en_num, string=text)
    return res


def Levenshtein_Distance(str1, str2):
    """
    calculate the Levenshtein distance between two strings
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def find_cache_bad_char():
    """
    Fix some weird char in CWQ.
    """
    data = read_json("dataset/complexwebquestions_V1_1/ComplexWebQuestions_train.json")
    data += read_json("dataset/complexwebquestions_V1_1/ComplexWebQuestions_dev.json")
    data += read_json("dataset/complexwebquestions_V1_1/ComplexWebQuestions_test.json")
    maps = {}
    for d in tqdm(data, desc="finding bad char"):
        q = d["question"]
        mids = re.findall(_pattern, d["sparql"].replace(")", " )").replace("}", " }"))
        mids = [m for m in set(mids) if len(m) > 2]
        ents = [fb.get_mid_name(mid) for mid in mids]
        ents = [n for n in ents if n]
        ent_words = []
        for ent in ents:
            ent_words += ent.split(" ")
        ent_words = sorted(set(ent_words))

        ques_list_sp = [i for i in q.split(" ") if find_special_char(i)]
        replaces = []
        for ques_sp in ques_list_sp:
            values = [Levenshtein_Distance(ques_sp, w) for w in ent_words]
            idx = _argmin(values)
            if idx > -1:
                replaces.append((ques_sp, ent_words[idx]))

        if replaces:
            maps[d["ID"]] = replaces
    save_to_json(maps, "dataset_processed/cwq/cwq_bad_char_map.json")


_FILTER_CLAUSE = 'FILTER ( NOT EXISTS {{ ?x ns:type.object.name ?xn . }} || EXISTS {{ ?x ns:type.object.name ?xn . FILTER ( ?xn != "{name}"@en ) }} )'


def replace_mid(text, _return_var_x_name=False, detail_mode=True):
    """
    use re to replace mid in SPAQRL. mid may start with "m." or "g."

    e.g.
    `ns:m.01xvb ns:sports.professional_sports_team.draft_picks ?y`
    ->
    `?e0 ns:type.object.name "xxx". ?e0 ns:sports.professional_sports_team.draft_picks ?y`


    detail_mode: add FILTER:
    e.g. FILTER ( NOT EXISTS { ?x ns:type.object.name ?xn . } || EXISTS { ?x ns:type.object.name ?xn . FILTER ( ?xn != "xxx"@en ) } )

    format:
        - one row per triple
        - the parentheses `{` and `}` are on their own line
        - filter is on its own line
    """
    # replace " {2,}ns:" with " ns:"
    text = re.sub(r" {2,}ns:", " ns:", text)
    text = text.replace("{", "\n{\n").replace("}", "\n}\n")
    lines = [l for l in text.split("\n") if l.strip()]
    new_lines = []
    depth = -1
    var_id = -1  # global var id
    mid_to_vare = {}
    last_head_entity = None
    is_in_semicolon = False
    var_x_n = None
    for line in lines:
        line = line.replace("\t", " ").replace("\n", " ").strip()
        if line.startswith("FILTER (?x != ?"):
            continue

        # add. ?x ns:type.object.name ?xn. FILTER (?xn !="xxx"@en)
        if detail_mode and line.startswith("FILTER (?x !="):
            var_x = line.split("FILTER (?x !=")[-1].split(")")[0].strip()
            var_x_n = fb.get_mid_name(var_x)
            new_line = _FILTER_CLAUSE.format(name=var_x_n)
            new_lines.append(new_line)
            continue
        if line.startswith("FILTER (!isLiteral(?x) OR"):
            continue

        # different depth should use different var
        if line == "{":
            depth += 1
            mid_to_vare[f"dep{depth}"] = {}
        elif line == "}":
            depth -= 1

        mid_res = re.findall(_pattern, line)
        if len(mid_res) > 0:
            for mid in mid_res:
                mid = mid.strip().split(")")[0]
                ename = fb.get_mid_name(mid)
                if mid not in mid_to_vare[f"dep{depth}"]:
                    var_id += 1
                    mid_to_vare[f"dep{depth}"][mid] = "?e{}".format(var_id)
                new_line = '?e{} ns:type.object.name "{}"@en .'.format(var_id, ename)
                new_lines.append(new_line)

        if depth >= 0:
            for mid, vare in mid_to_vare[f"dep{depth}"].items():
                line = line.replace(mid, vare)

        # add head entity
        if " ;" in line:
            line = line.replace(" ;", " .")
            # fisrt line
            if not is_in_semicolon:
                is_in_semicolon = True
                last_head_entity = line.split(" ")[0]
            # other line
            else:
                assert last_head_entity
                line = last_head_entity + " " + line
        # last line
        elif is_in_semicolon:
            is_in_semicolon = False
            assert last_head_entity
            line = last_head_entity + " " + line
            last_head_entity = None

        new_lines.append(line)

    sparql = "\n".join(new_lines)
    sparql = sparql.replace("ns:?", "?")

    # check
    mid_res = re.findall(_pattern, sparql)
    assert len(mid_res) == 0

    if _return_var_x_name:
        return sparql, var_x_n
    return sparql


def _add_num_postfix(sparql):
    """
    e.g. ?c ns:measurement_unit.dated_integer.number "11179" . -> ?c ns:measurement_unit.dated_integer.number "11179"^^xsd:float .
    """

    def is_float(value):
        value = value.replace('"', "").strip()
        try:
            float(value)
            return True
        except ValueError:
            return False

    # one line: ?x ns:geography.mountain.prominence \"1107.0\" .
    # if line[-2] is float:
    rep_map = {}
    for line in sparql.split("\n"):
        if "ns:" in line:
            line = line.split(" ")
            if len(line) > 3 and is_float(line[-2]):
                rep_map[line[-2]] = line[-2] + "^^xsd:float"

    if rep_map:
        print(rep_map)
    for src, to in rep_map.items():
        sparql = sparql.replace(src, to)
    return sparql


def _single(d, maps):
    if "#MANUAL" in d["sparql"]:
        return None

    for k in [
        "composition_answer",
        "created",
        "machine_question",
        "webqsp_ID",
    ]:
        d.pop(k, None)

    sparql = replace_mid(d["sparql"])

    # try to add xsd:float if ans is empty.
    ans = ExecuteSPARQL(sparql, str_mode=False)
    if not ans:
        sparql = _add_num_postfix(sparql)
        ans = ExecuteSPARQL(sparql, str_mode=False)
        if not ans:
            return None

    d["raw_sparql"] = d.pop("sparql")
    d["sparql"] = sparql
    if "answers" in d:
        d["answers_raw"] = d.pop("answers")
    d["answers"] = ans

    q = d["question"]
    d["char_map"] = []
    _id = d.pop("ID")
    d["id"] = _id
    if _id in maps:
        d["char_map"] = maps[_id]
        for bad_str, good_str in maps[_id]:
            q = q.replace(bad_str, good_str)
    d["question"] = q
    return d


def split_data_by_type(split="train"):
    """
    types: {'composition', 'conjunction','comparative', 'superlative'}
    save to: dataset_processed/cwq/train/[type].json

    - train: each type 100.
    - dev: each type 100.
    """
    data = read_json(f"dataset/complexwebquestions_V1_1/ComplexWebQuestions_{split}.json")

    # debug
    # data = [d for d in data if d["ID"]=="WebQTrn-1519_f2b80b8985be7c3bdcb47a351a865f3e"]

    # {"WebQTrn-0": [("aa", "bb")], ...}
    maps = read_json("dataset/complexwebquestions_V1_1/cwq_bad_char_map.json")
    for _type in ["composition", "conjunction", "comparative", "superlative"]:
        type_data = [d for d in data if d["compositionality_type"] == _type]
        random.seed(123)
        random.shuffle(type_data)
        res = multi_process(
            items=type_data[:200],
            process_function=_single,
            cpu_num=os.cpu_count() - 1,
            debug=False,
            dummy=True,
            maps=maps,
        )
        res = [i for i in res if i][:150]
        save_to_json(res, f"dataset_processed/cwq/{split}/{_type}.json")



if __name__ == "__main__":
    """
    python data_preprocess/cwq.py
    """
    find_cache_bad_char()

    split_data_by_type("dev")
    split_data_by_type("test")
