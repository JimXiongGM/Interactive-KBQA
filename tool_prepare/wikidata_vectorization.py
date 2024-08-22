import os
import pickle
from traceback import print_exc

import chromadb
from tqdm import tqdm

from common.common_utils import read_json, read_jsonl, save_to_pkl
from tool.openai_api import get_embedding, get_embedding_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


node_vec_pkl_file = "database/cache_vector_db_kqapro/node-name.pkl"


def cache_node_vec():
    if os.path.exists(node_vec_pkl_file):
        return
    data = read_jsonl("database/wikidata-kqapro-info/node-name-desc.jsonl")
    batch_data = []  # [{"id": Qxx, "text": "A xx node, xxx"}, ...]
    for d in data:
        # desc is not used.
        name, tp, desc = d["name"], d["type"], d["desc"]
        _a = "an" if tp == "entity" else "a"

        # i.e. "a entity about xxx" or "an concept about xxx"
        _text = f"{_a} {tp} about {name}"

        item = {}
        item["id"] = d["id"]
        item["text"] = _text
        batch_data.append(item)

    id_vecs = {}
    texts = [i["text"] for i in batch_data]
    ids = [i["id"] for i in batch_data]

    batch_size = 500  # Define your batch size here
    for batch_texts, batch_ids in tqdm(
        zip(chunks(texts, batch_size), chunks(ids, batch_size)),
        desc="vec batch",
        total=len(texts) // batch_size + 1,
    ):
        batch_vec = get_embedding_batch(batch_texts)
        assert len(batch_vec) == len(batch_texts)
        for _id, vec in zip(batch_ids, batch_vec):
            id_vecs[_id] = vec
    save_to_pkl(id_vecs, node_vec_pkl_file)


def cache_kqapro_attr_rela_qual_vectors(mode="attribute", vectorization="openai"):
    assert mode in ["attribute", "relation", "qualifier"], "mode error"

    data = read_json(f"database/wikidata-kqapro-info/kqapro_{mode}s_counter.json")
    out_f = f"database/cache_vector_db_kqapro/kqapro-{mode}s-{vectorization}.pkl"

    id_vecs = {}
    if os.path.exists(out_f):
        id_vecs = pickle.load(open(out_f, "rb"))
    for name in tqdm(data.keys(), ncols=100, desc=mode):
        if name in id_vecs:
            continue
        try:
            vec = get_embedding(name)
            id_vecs[name] = vec
        except:
            print_exc()
            break
    save_to_pkl(id_vecs, out_f)


client = None


def init_chroma_client():
    global client
    if client is None:
        client = chromadb.PersistentClient(path="./database/db_vector_chroma_v2")
    return client


# vectorization node / edge
def vectorization_chroma_nodes():
    """
    return item: {"name": xxx, "type": "person", "desc": xxx}
    """
    init_chroma_client()
    mix_info_map = {}

    data = read_jsonl("database/wikidata-kqapro-info/node-name-desc.jsonl")
    id_vecs = pickle.load(open(node_vec_pkl_file, "rb"))

    # entity
    entities = [d for d in data if d["type"] == "entity"]
    for d in entities:
        name = d["name"]
        if name in mix_info_map:
            mix_info_map[name]["valid_types"].append("entity")
        else:
            mix_info_map[name] = {
                "name": name,
                "embedding": id_vecs[d["id"]],
                "id": str(len(mix_info_map)),
                "valid_types": ["entity"],
                "desc": d["desc"],
            }

    # concept
    concepts = [d for d in data if d["type"] == "concept"]
    for d in concepts:
        name = d["name"]
        if name in mix_info_map:
            mix_info_map[name]["valid_types"].append("concept")
            # 和entity重名，desc增加特殊符号
            mix_info_map[name]["desc"] += " [DESC] " + d["desc"]
            print(f"concept: `{name}` has same name with entity")
        else:
            mix_info_map[name] = {
                "name": name,
                "embedding": id_vecs[d["id"]],
                "id": str(len(mix_info_map)),
                "valid_types": ["concept"],
                "desc": d["desc"],
            }

    # to chroma data type
    ids, embeddings, metadatas, documents = [], [], [], []
    for name, info in mix_info_map.items():
        ids.append(info["id"])
        embeddings.append(info["embedding"])
        _t = ", ".join(set(info["valid_types"]))
        assert _t
        info["desc"] = info["desc"] if info["desc"] else ""
        metadatas.append({"valid_types": _t, "desc": info["desc"]})
        documents.append(info["name"])

    # chroma
    index_name = "node-onlyname"
    names = [i.name for i in client.list_collections()]
    if index_name in names:
        client.delete_collection(index_name)
    collection = client.create_collection(index_name)
    collection.add(
        embeddings=embeddings,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    count = collection.count()

    print(f"{index_name}: {count}")


def vectorization_chroma_edges():
    mix_info_map = {}

    # attribute
    id_vecs = pickle.load(open("database/cache_vector_db_kqapro/kqapro-attributes-openai.pkl", "rb"))
    data: dict = read_json("database/wikidata-kqapro-info/kqapro_attributes_counter.json")
    for name in data:
        if name in mix_info_map:
            mix_info_map[name]["valid_types"].append("attribute")
        else:
            mix_info_map[name] = {
                "name": name,
                "embedding": id_vecs[name],
                "id": str(len(mix_info_map)),
                "valid_types": ["attribute"],
            }

    # relation
    id_vecs = pickle.load(open("database/cache_vector_db_kqapro/kqapro-relations-openai.pkl", "rb"))
    data: dict = read_json("database/wikidata-kqapro-info/kqapro_relations_counter.json")
    for name in data:
        if name in mix_info_map:
            mix_info_map[name]["valid_types"].append("relation")
        else:
            mix_info_map[name] = {
                "name": name,
                "embedding": id_vecs[name],
                "id": str(len(mix_info_map)),
                "valid_types": ["relation"],
            }

    # qualifier
    id_vecs = pickle.load(open("database/cache_vector_db_kqapro/kqapro-qualifiers-openai.pkl", "rb"))
    data: dict = read_json("database/wikidata-kqapro-info/kqapro_qualifiers_counter.json")
    for name in data:
        if name in mix_info_map:
            mix_info_map[name]["valid_types"].append("qualifier")
        else:
            mix_info_map[name] = {
                "name": name,
                "embedding": id_vecs[name],
                "id": str(len(mix_info_map)),
                "valid_types": ["qualifier"],
            }

    # to chroma data type
    ids, embeddings, metadatas, documents = [], [], [], []
    for name, info in mix_info_map.items():
        ids.append(info["id"])
        embeddings.append(info["embedding"])
        _t = ", ".join(set(info["valid_types"]))
        assert _t
        metadatas.append({"valid_types": _t})
        documents.append(info["name"])

    # chroma
    index_name = "kqapro-db-edges"
    names = [i.name for i in client.list_collections()]
    if index_name in names:
        client.delete_collection(index_name)
    collection = client.create_collection(index_name)
    collection.add(
        embeddings=embeddings,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    count = collection.count()

    # kqapro-db-all-elements: 15412
    print(f"{index_name}: {count}")


collection_nodes = None


def SearchName(query, size=10):
    global collection_nodes
    if collection_nodes is None:
        client = init_chroma_client()
        collection_nodes = client.get_collection(name="node-onlyname")

    result = []
    q_emb = get_embedding(query.strip())
    result = collection_nodes.query(
        query_embeddings=q_emb,
        n_results=size,
    )

    # return: [{"name": xxx, "type": xxx, "desc": xxx, "distance": xxx}, ...]
    format_results = []
    visited = set()  # name+type
    for metadata, document, distance in zip(
        result["metadatas"][0], result["documents"][0], result["distances"][0]
    ):
        _k = document + metadata["valid_types"]
        if _k in visited:
            continue
        visited.add(_k)
        if document.lower() == query.lower():
            distance = 0
        format_results.append(
            {
                "name": document,
                "type": metadata["valid_types"],
                "distance": distance,
            }
        )
    format_results.sort(key=lambda x: x["distance"])
    return format_results[:size]


if __name__ == "__main__":
    cache_node_vec()
    cache_kqapro_attr_rela_qual_vectors(mode="attribute")
    cache_kqapro_attr_rela_qual_vectors(mode="relation")
    cache_kqapro_attr_rela_qual_vectors(mode="qualifier")

    vectorization_chroma_nodes()
    vectorization_chroma_edges()

    r = SearchName("art piece")
    print(r)

    r = SearchName("metro borough")
    print(r)
