import gzip
import os
import re
import string

from tqdm import tqdm

from common.common_utils import save_to_json

"""
Require:
    - freebase-literal_fixed-lan.gz
Output:
    - freebase_entity_name_en.txt
"""

name_map = {}
with gzip.open("database/intermediate_file/fb_filter_eng_fix_literal.gz") as f1:
    pbar = tqdm(f1, total=955648474)
    for idx, line in enumerate(pbar):
        line = line.decode("utf-8").strip().split("\t")
        mid = line[0].replace("<http://rdf.freebase.com/ns/", "").replace(">", "")
        if mid[:2] not in ["m.", "g."]:
            continue
        if line[1] == "<http://rdf.freebase.com/ns/type.object.name>":
            name_map[mid] = line[2]

save_to_json(name_map, "database/freebase-info/freebase_entity_name.json")

names = sorted(set(name_map.values()))


def has_one_enchar(text):
    text = text.replace(" ", "")
    for c in text:
        if c.encode("utf-8").isalpha():
            return True
    return False


def is_full_width(char):
    unicode_val = ord(char)
    if (unicode_val >= 65281 and unicode_val <= 65374) or (unicode_val >= 0xFF01 and unicode_val <= 0xFF5E):
        return True
    return False


remove_nota = "[’·°–!\"#$%&'()*+,-./:;<=>?@，。★、…【】（）《》？“”‘’！[\\]^_`{|}~]+"
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def remove_punctuation(text):
    text = re.sub(remove_nota, "", text)
    text = text.translate(remove_punctuation_map)
    return text


def remove_number(text):
    text = re.sub("[0-9]", "", text)
    return text


def non_en_char_ratio(text):
    text = text.replace(" ", "")
    if len(text) == 0:
        return 0
    cnt = 0
    for c in text:
        if not c.encode("utf-8").isalpha():
            cnt += 1
    return cnt / len(text)


def _is_bad(n):
    n = n.strip('"').strip()
    n = n.replace(" ", "")

    if len(n) == 0:
        return True, 1

    if n.startswith("#") and len(n) == 33:
        return True, 2

    # #10983 started with "#" and all numbers
    if n.startswith("#") and n[1:].isdigit():
        return True, 3

    # all are symbols
    if all([x in ".,:;?!@#$%^&*()_+-=<>{}[]|\/" for x in n]):
        return True, 4

    if len(n) == 1 or len(n) > 100:
        return True, 5

    if "�" in n or "" in n:
        return True, 6

    n1 = remove_punctuation(n)
    n1 = remove_number(n1)

    if non_en_char_ratio(n1) > 0.4:
        return True, 7

    if any([is_full_width(c) for c in n1]):
        return True, 8

    return False, 0


_good = []
for n in names:
    n = n.replace("@en", "")
    _c, _number = _is_bad(n)
    if not _c:
        _good.append(n)
len(_good)

# save to txt
os.makedirs("database/freebase-info", exist_ok=True)
with open("database/freebase-info/freebase_entity_name_en.txt", "w") as f1:
    f1.write("\n".join(_good))
