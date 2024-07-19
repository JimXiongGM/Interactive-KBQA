import pickle

from loguru import logger

from common.common_utils import save_to_pkl
from tool.openai import get_embedding_batch


def cache_predicate_vectors_batch():
    out_f = f"database/cache_vector_db_metaqa/predicate-openai-split-to-words.pkl"

    preds = {
        "directed_by",
        "has_genre",
        "has_imdb_rating",
        "has_imdb_votes",
        "has_tags",
        "in_language",
        "release_year",
        "starred_actors",
        "written_by",
    }
    _preds = [p.replace("_", " ") for p in preds]

    id_vecs = {}
    batch_vec = get_embedding_batch(_preds)
    assert len(batch_vec) == len(preds)
    for _p, vec in zip(preds, batch_vec):
        id_vecs[_p] = vec

    save_to_pkl(id_vecs, out_f)


def get_chroma_client():
    import chromadb

    vec_client = chromadb.PersistentClient(path="./database/db_vector_chroma_metaqa_v1")
    return vec_client


vec_client = get_chroma_client()
CHROMA_PREDICATE_NAME = "predicate-openai-split-to-words"


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
    """
    python tool_prepare/metaqa.py
    """
    cache_predicate_vectors_batch()
    vectorization_chroma_predicate(
        name=CHROMA_PREDICATE_NAME,
        vec_f="database/cache_vector_db_metaqa/predicate-openai-split-to-words.pkl",
    )
