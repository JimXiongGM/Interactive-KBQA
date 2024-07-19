from typing import List

from tqdm import tqdm

from tool.client_es import ESClient

FB_MID_INDEX = "freebase-db-mid-name-en-v3"

es_client = None


def create_mid():
    global es_client
    if es_client is None:
        es_client = ESClient()
    index = FB_MID_INDEX

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
        }
    }

    es_client.delete_index(index)
    es_client.create_index(index, mappings, settings=settings)
    _file = open("database/freebase-info/freebase_entity_name_en.txt")
    actions = []
    for idx, d in enumerate(tqdm(_file, ncols=100)):
        d = d.strip().strip('"')
        actions.append(
            {
                "_id": idx,
                "_index": index,
                "name": d,
            }
        )
    es_client.bulk_insert(actions)
    _file.close()
    print(es_client.search(index, query={"match": {"name": "obama"}}))


def SearchMidName(query, size=5) -> List[str]:
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
    index = FB_MID_INDEX

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
    names = [i["_source"]["name"] for i in res["hits"]["hits"]]

    # wierd. ES will not rank the exact match to the top
    _head, _other = [], []
    for n in names:
        if n.lower() == query.lower():
            _head.append(n)
        else:
            _other.append(n)
    names = _head + _other
    return names[:size]


def check_total_num():
    es_client = ESClient()
    res = es_client.count(FB_MID_INDEX)
    print(res)


if __name__ == "__main__":
    # create_mid()

    x = SearchMidName("Southern Peninsula")
    print(x)

    check_total_num()
