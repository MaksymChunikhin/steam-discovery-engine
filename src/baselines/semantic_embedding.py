import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from .base import BaseRecommender


class SemanticEmbeddingRecommender(BaseRecommender):
    # Embedding retrieval: rank games by cosine similarity between a user profile
    # and precomputed semantic game embeddings (from review text).
    # Embedding retrieval: ранжируем игры по cosine между профилем юзера и
    # предрассчитанными семантическими эмбеддингами игр (из текста отзывов).

    def __init__(self, game_embeddings: np.ndarray, game_ids,
                 user_col: str = "user_id", item_col: str = "game_id",
                 batch_size: int = 5000):
        self.item_vecs_ = np.asarray(game_embeddings, dtype=np.float32)
        self.item_ids_ = np.asarray(game_ids)
        self.item_index_ = {g: i for i, g in enumerate(self.item_ids_)}
        self.user_col = user_col
        self.item_col = item_col
        self.batch_size = batch_size

    def fit(self, train: pd.DataFrame) -> "SemanticEmbeddingRecommender":
        # Keep only train rows whose game has an embedding
        # Оставляем только train-строки, у игр которых есть эмбеддинг
        t = train[train[self.item_col].isin(self.item_index_)]
        self.user_ids_ = t[self.user_col].unique()
        self.user_index_ = {u: i for i, u in enumerate(self.user_ids_)}

        rows = t[self.user_col].map(self.user_index_).to_numpy()
        cols = t[self.item_col].map(self.item_index_).to_numpy()
        incidence = sparse.csr_matrix(
            (np.ones(len(t), dtype=np.float32), (rows, cols)),
            shape=(len(self.user_ids_), len(self.item_ids_)),
        )

        # User profile = L2-normalized mean of their games' embeddings (simple mean)
        # Профиль юзера = L2-нормированное среднее эмбеддингов его игр (простое среднее)
        profiles = incidence @ self.item_vecs_
        self.user_profiles_ = np.asarray(normalize(profiles), dtype=np.float32)

        self.seen_idx_ = (
            pd.Series(cols, index=t[self.user_col].to_numpy())
            .groupby(level=0).agg(list).to_dict()
        )
        return self

    def recommend(self, user_ids, k: int = 10) -> dict:
        recs = {}
        user_ids = list(user_ids)
        for start in range(0, len(user_ids), self.batch_size):
            batch = user_ids[start:start + self.batch_size]
            valid = [u for u in batch if u in self.user_index_]
            for u in batch:
                if u not in self.user_index_:
                    recs[u] = []
            if not valid:
                continue
            rows = [self.user_index_[u] for u in valid]
            scores = self.user_profiles_[rows] @ self.item_vecs_.T  # (b x n_items)
            for i, u in enumerate(valid):
                row = scores[i]
                seen = self.seen_idx_.get(u)
                if seen is not None:
                    row[seen] = -np.inf
                top = np.argpartition(-row, k)[:k]
                top = top[np.argsort(-row[top])]
                recs[u] = [self.item_ids_[j] for j in top]
        return recs
