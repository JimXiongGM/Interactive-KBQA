import pickle

from loguru import logger
from tqdm import tqdm

from common.common_utils import read_json, read_jsonl, save_to_pkl
from tool.openai import get_embedding_batch


def cache_predicate_vectors_batch():
    """
    database/freebase-info/predicate_freq.json
    """
    data = read_json("database/freebase-info/predicate_freq.json")
    out_f = f"database/cache_vector_db_fb/predicate-openai-split-to-words.pkl"

    preds = list(data.keys())
    preds = [
        p[1:-1].replace("http://rdf.freebase.com/ns/", "") for p in preds if p[0] == "<" and p[-1] == ">"
    ]
    preds = [p.replace(".", " . ").replace("_", " _ ") for p in preds]
    logger.info(f"Total {len(preds)} predicates")

    id_vecs = {}
    for batch_preds in tqdm(
        [preds[i : i + 1000] for i in range(0, len(preds), 1000)],
        desc="cache predicate vectors",
    ):
        _batch_preds = [p.replace(" . ", ".").replace(" _ ", "_") for p in batch_preds]
        batch_vec = get_embedding_batch(_batch_preds)
        assert len(batch_vec) == len(batch_preds)
        for _p, vec in zip(batch_preds, batch_vec):
            id_vecs[_p] = vec

    save_to_pkl(id_vecs, out_f)


def cache_cvt_predicate_vectors_batch():
    """
    from: database/freebase-info/cvt_predicate_onehop.jsonl
    save to: database/cache_vector_db_fb/cvt-predicate-pair-openai-split-to-words.pkl
    """
    data = read_jsonl("database/freebase-info/cvt_predicate_onehop.jsonl")

    data = {d["key"]: d["value"] for d in data}
    pred_pair = [f"{p1} -> {p2}" for p1, v in data.items() for p2 in v]
    logger.info(f"Total {len(pred_pair)} pred_pair")

    out_f = f"database/cache_vector_db_fb/cvt-predicate-pair-openai-split-to-words.pkl"
    id_vecs = {}
    for batch_pairs in tqdm(
        [pred_pair[i : i + 1000] for i in range(0, len(pred_pair), 1000)],
        desc="cache predicate vectors",
    ):
        _batch_pairs = [p.replace(".", " . ").replace("_", " _ ") for p in batch_pairs]
        batch_vec = get_embedding_batch(_batch_pairs)
        assert len(batch_vec) == len(batch_pairs)
        for _p, vec in zip(batch_pairs, batch_vec):
            id_vecs[_p] = vec

    save_to_pkl(id_vecs, out_f)


# ----------------------------------------------------------------------- #


def get_chroma_client():
    import chromadb

    vec_client = chromadb.PersistentClient(path="./database/db_vector_chroma_fb_v2")
    return vec_client


vec_client = get_chroma_client()

CHROMA_PREDICATE_NAME = "predicate-openai-split-to-words"
CHROMA_CVT_PREDICATE_PAIR_NAME = "cvt-predicate-pair-openai-split-to-words"


def vectorization_chroma_predicate(name, vec_f):
    id_vecs = pickle.load(open(vec_f, "rb"))

    embeddings, ids, documents = [], [], []
    for idx, (p, vec) in enumerate(id_vecs.items()):
        embeddings.append(vec)
        ids.append(str(idx))
        documents.append(p)

    names = [i.name for i in vec_client.list_collections()]
    if name in names:
        vec_client.delete_collection(name)
    collection = vec_client.create_collection(name)
    collection.add(embeddings=embeddings, ids=ids, documents=documents)
    count = collection.count()
    logger.info(f"{name}: {count}")


if __name__ == "__main__":
    cache_predicate_vectors_batch()
    vectorization_chroma_predicate(
        name=CHROMA_PREDICATE_NAME,
        vec_f="database/cache_vector_db_fb/predicate-openai-split-to-words.pkl",
    )

    cache_cvt_predicate_vectors_batch()
    vectorization_chroma_predicate(
        name=CHROMA_CVT_PREDICATE_PAIR_NAME,
        vec_f="database/cache_vector_db_fb/cvt-predicate-pair-openai-split-to-words.pkl",
    )
