import json
import os

from common.common_utils import jsonl_generator, multi_process, read_json
from tool.client_wikidata_kqapro import WikidataKQAProClient

"""
This file cache desc of entities and concepts.
But the desc is not used in the project. We keep it here for future use.
"""
# os.environ["http_proxy"] = "http://127.0.0.1:7893"
# os.environ["https_proxy"] = "http://127.0.0.1:7893"
wd = WikidataKQAProClient(end_point="https://query.wikidata.org/sparql")


def _single(d):
    try:
        d["desc"] = wd.get_qid_desc(d["id"])
        return d
    except:
        return None


def cache_entity_info():
    """
    kb.json has two kinds of nodes: ['concepts', 'entities']
    but dont has the desc of the nodes.
    """
    kb = read_json("dataset/KQA-Pro-v1.0/kb.json")

    cache_file = f"database/wikidata-kqapro-info/node-name-desc.jsonl"
    skip_ids = []
    if os.path.exists(cache_file):
        skip_ids = [i["id"] for i in jsonl_generator(cache_file)]
        skip_ids = set(skip_ids)
        print(f"Skip id: {len(skip_ids)}")

    def _gen():
        # "Q123": {['name', 'instanceOf', 'attributes', 'relations']}
        # len: 16960
        for qid, item in kb["entities"].items():
            if qid in skip_ids:
                continue
            d = {}
            d["id"] = qid
            d["name"] = item["name"]
            d["type"] = "entity"
            yield d

        # "Q123": {['name', 'instanceOf']}
        # len: 794
        for qid, item in kb["concepts"].items():
            if qid in skip_ids:
                continue
            d = {}
            d["id"] = qid
            d["name"] = item["name"]
            d["type"] = "concept"
            yield d

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    out_file = open(cache_file, "a", encoding="utf-8")
    _save = lambda x: out_file.write(json.dumps(x, ensure_ascii=False) + "\n") if x else 0

    multi_process(
        items=_gen(),
        process_function=_single,
        postprocess_function=_save,
        cpu_num=100,
        dummy=True,
    )
    out_file.close()


if __name__ == "__main__":
    cache_entity_info()
