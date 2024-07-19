from typing import Dict, List

from tqdm import tqdm

from tool.client_es import ESClient

METAQA_INDEX = "metaqa-db-name-v1"

es_client = None


def _load_entity():
    kb = []
    with open("./dataset/MetaQA-vanilla/kb.txt", "r") as f:
        for line in f:
            line = line.strip().split("|")
            assert len(line) == 3
            kb.append(line)
    for h, r, t in kb:
        if r in ["directed_by", "written_by", "starred_actors"]:
            t_type = "person"
            h_type = "movie"
            yield h, h_type
            yield t, t_type


def create_index():
    global es_client
    if es_client is None:
        es_client = ESClient()
    index = METAQA_INDEX

    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "std_folded": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"],
                }
            }
        },
    }

    mappings = {
        "properties": {
            "name": {
                "type": "text",
                "analyzer": "std_folded",
                "search_analyzer": "std_folded",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "type": {"type": "keyword"},
        }
    }

    es_client.delete_index(index)
    es_client.create_index(index, mappings, settings=settings)
    visited = set()
    actions = []
    for idx, (ent, ent_type) in enumerate(tqdm(_load_entity(), ncols=100)):
        _k = ent + ent_type
        if _k in visited:
            continue
        actions.append(
            {
                "_id": idx,
                "_index": index,
                "name": ent,
                "type": ent_type,
            }
        )
        visited.add(_k)

    # add tag
    # The dataset is quite peculiar. The tag should be a literal, but it appears as a start node in the query. The tag must be searchable.
    visited = set()
    with open("./dataset/MetaQA-vanilla/kb.txt", "r") as f:
        for line in f:
            line = line.strip().split("|")
            assert len(line) == 3
            h, r, t = line
            if r == "has_tags":
                _k = t + "tag"
                if _k in visited:
                    continue
                visited.add(_k)
                idx += 1
                actions.append(
                    {
                        "_id": idx,
                        "_index": index,
                        "name": t,
                        "type": "tag",
                    }
                )

    es_client.bulk_insert(actions)

    print(es_client.search(index, query={"match": {"name": "obama"}}))


def SearchMidName(query, size=5) -> List[Dict[str, str]]:
    """
    fuzzy query:
    GET index_name/_search
    {
    "query": {
        "match": {
        "name": {
            "query": "elastic search",
            "fuzziness": 0,
            "prefix_length": 0,
            "max_expansions": 50,
            "transpositions": true
        }
        }
    }
    }
    fuzziness: 0 means exact match, 1 means 1 edit distance, 2 means 2 edit distance.
    prefix_length: 0 means no prefix, 1 means 1 char prefix, 2 means 2 char prefix.
    max_expansions: max number of terms to match.
    transpositions: whether to allow transpositions. eg. "ab" -> "ba"
    """
    global es_client
    if es_client is None:
        es_client = ESClient()
    index = METAQA_INDEX
    # query_dsl = {"match": {"name": str(query)}}
    query_dsl = {
        "match": {
            "name": {
                "query": str(query),
                "fuzziness": 2,
                "prefix_length": 5,
            }
        }
    }

    res = es_client.search(index, query=query_dsl, size=100)

    name_types = [i["_source"] for i in res["hits"]["hits"]]

    # wierd. ES will not rank the exact match to the top
    _head, _other = [], []
    for n_t in name_types:
        if n_t["name"] == query:
            _head.append(n_t)
        else:
            _other.append(n_t)
    _name_types = _head + _other
    return _name_types[:size]


if __name__ == "__main__":
    create_index()

    x = SearchMidName("ginger rogers")
    print(x)
