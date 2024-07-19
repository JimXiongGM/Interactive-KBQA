import os
import random

from common.common_utils import save_to_json

"""
metaqa
url: https://github.com/yuyuz/MetaQA
"""


def load_entity_map():
    visited = set()
    entity_to_id = {}
    for line in open("dataset/MetaQA-vanilla/entity/kb_entity_dict.txt"):
        line = line.strip().split("\t")
        assert len(line) == 2
        idx, entity = line
        if entity in visited:
            continue
        visited.add(entity)
        entity_to_id[entity] = f"e{idx}"
    print(f"load {len(entity_to_id)} entities")
    return entity_to_id


def make_nt_file():
    """
    schema:

    directed_by| movie | person
    written_by | movie | person
    starred_actors | movie | person
    release_year | movie | year
    in_language | movie | string
    has_tags | movie | string
    has_genre | movie | string
    has_imdb_votes | movie | string
    has_imdb_rating | movie | string

    <http://metaqa/movie/xxx>
    <http://metaqa/person/xxx>
    "1980"^^<http://www.w3.org/2001/XMLSchema#gYear>
    """
    kb = []
    with open("./dataset/MetaQA-vanilla/kb.txt", "r") as f:
        for line in f:
            line = line.strip().split("|")
            assert len(line) == 3
            kb.append(line)

    entity_to_id = load_entity_map()

    os.makedirs("./database/MetaQA-vanilla", exist_ok=True)
    nodes = []
    with open("./database/MetaQA-vanilla/kb.nt", "w", encoding="utf-8") as f:
        # according to the schema.
        for h, r, t in kb:
            # t must be entity
            if r in ["directed_by", "written_by", "starred_actors"]:
                t = "<" + entity_to_id[t] + ">"
                nodes.append(t)

            # typed-literal
            elif r == "release_year":
                t = f'"{t}"^^<http://www.w3.org/2001/XMLSchema#gYear>'

            # literal
            elif r in [
                "in_language",
                "has_tags",
                "has_genre",
                "has_imdb_votes",
                "has_imdb_rating",
            ]:
                t = f'"{t}"'
            else:
                raise ValueError(f"unknown relation {r}")

            # h must be entity
            h = "<" + entity_to_id[h] + ">"
            nodes.append(h)

            f.write(f"{h} <{r}> {t} .\n")

        # write entity name
        for entity, entity_id in entity_to_id.items():
            f.write(f'<{entity_id}> <name> "{entity}" .\n')
    print("make nt file done")
    nodes = set(nodes)
    print(f"len(nodes): {len(nodes)}")


def split_data_by_type(split="test"):
    assert split in ["train", "test"]
    for _type in ["1-hop", "2-hop", "3-hop"]:
        with open(f"./dataset/MetaQA-vanilla/{_type}/qa_{split}_qtype.txt", "r") as typef, open(
            f"./dataset/MetaQA-vanilla/{_type}/vanilla/qa_{split}.txt", "r"
        ) as dataf:
            infer_chains = [i.strip() for i in typef.readlines()]
            data_lines = [i.strip() for i in dataf.readlines()]
        assert len(infer_chains) == len(data_lines)

        # to json format
        data = []
        for idx, (infer_chain, line) in enumerate(zip(infer_chains, data_lines)):
            raw_q, ans = line.split("\t")
            ans = ans.split("|")
            # clean [ and ]
            q = raw_q.replace("[", "").replace("]", "").strip()
            item = {
                "id": f"{_type}-{idx}",
                "infer_chain": infer_chain,
                "question": q,
                "raw_q": raw_q,
                "answer": ans,
            }
            data.append(item)

        random.seed(123)
        random.shuffle(data)
        data = data[:300]
        print(f"len(data): {len(data)}")
        save_to_json(data, f"./dataset_processed/metaqa/{split}/{_type}.json")


if __name__ == "__main__":
    """
    python data_preprocess/metaqa.py
    """
    make_nt_file()
    split_data_by_type(split="test")
