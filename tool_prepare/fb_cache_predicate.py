import gzip
import json
import os
from collections import defaultdict

from loguru import logger
from tqdm import tqdm

from common.common_utils import jsonl_generator, multi_process, save_to_json
from tool.client_freebase import FreebaseClient

"""
Require:
    - freebase-literal_fixed-lan.gz
    - fb_properties_expecting_cvt.txt; adapted from: https://github.com/google/freebase-wikidata-converter/blob/master/dataset_processed/properties_expecting_cvt.csv
Output:
    - predicate_freq.json
    - cvt_predicate_onehop.jsonl
"""


def cache_predicate_in_fb():
    """
    cache all predicate in freebase.
    """
    pred_count = defaultdict(int)
    with gzip.open("database/intermediate_file/fb_filter_eng_fix_literal.gz") as f1:
        for line in tqdm(f1, total=955648474):
            line = line.decode("utf-8").strip().split("\t")
            if len(line) != 4:
                continue
            subj, pred, obj = line
            if subj.startswith("<http://rdf.freebase.com/ns/g.") or subj.startswith(
                "<http://rdf.freebase.com/ns/m."
            ):
                if pred.startswith("<http://rdf.freebase.com/ns") and pred[-1] == ">":
                    pred_count[pred] += 1

    # sort by frequency
    pred_count = sorted(pred_count.items(), key=lambda x: x[1], reverse=True)

    # save predicate_freq.json
    pred_count = {k: v for k, v in pred_count}
    save_to_json(pred_count, "database/freebase-info/predicate_freq.json")


def single_for_cvt(p, fb: FreebaseClient):
    sparql = f"PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?_pout WHERE {{ ?_e ns:{p} ?_cvt . ?_cvt ?_pout ?_t . }}"
    query_res = fb.query(sparql)
    pouts = [i["_pout"]["value"].replace("http://rdf.freebase.com/ns/", "") for i in query_res]
    return {"key": p, "value": pouts}


def cache_cvt_predicate_pair_in_fb(debug=True, cpu_num=1):
    """
    cache all predicate pairs in freebase.
    e.g. people.person.place_of_birth -> location.location.containedby
    """
    data = open("tool_prepare/fb_properties_expecting_cvt.txt").readlines()
    data = [d.strip() for d in data]

    out_name = "database/freebase-info/cvt_predicate_onehop.jsonl"
    if os.path.exists(out_name):
        skip_preds = [i["key"] for i in jsonl_generator(out_name) if i]
        skip_preds = set(skip_preds)
        logger.info(f"Skip key: {len(skip_preds)}")
        data = [d for d in data if d not in skip_preds]

    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    out_file = open(out_name, "a", encoding="utf-8")
    _save = lambda x: out_file.write(json.dumps(x, ensure_ascii=False) + "\n") if x else None
    fb = FreebaseClient()
    multi_process(
        items=data,
        process_function=single_for_cvt,
        postprocess_function=_save,
        cpu_num=cpu_num,
        debug=debug,
        dummy=True,
        fb=fb,
    )
    out_file.close()


if __name__ == "__main__":
    cache_predicate_in_fb()
    cache_cvt_predicate_pair_in_fb(debug=False, cpu_num=os.cpu_count())
