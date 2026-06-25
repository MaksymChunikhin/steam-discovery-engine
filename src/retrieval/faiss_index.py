import faiss
import numpy as np


def build_index(embeddings: np.ndarray):
    # Flat inner-product index; with L2-normalized vectors inner product = cosine.
    # Плоский inner-product индекс; на L2-нормированных векторах IP = cosine.
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def search(index, queries: np.ndarray, k: int):
    # Search the index for the top-k nearest items per query (cosine)
    # Ищем top-k ближайших айтемов на запрос (cosine)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    faiss.normalize_L2(q)
    return index.search(q, k)
