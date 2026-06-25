import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from .base import BaseRecommender
from ..features import split_genres, split_tags


class ContentBasedRecommender(BaseRecommender):
    # Classical vector-space retrieval: multi-hot (genres * weight + tags) -> cosine.
    # Классический vector-space retrieval: multi-hot (жанры * вес + теги) -> cosine.

    def __init__(self, games: pd.DataFrame, genre_weight: int = 2,
                 user_col: str = "user_id", item_col: str = "game_id",
                 batch_size: int = 5000):
        self.games = games
        self.genre_weight = genre_weight
        self.user_col = user_col
        self.item_col = item_col
        self.batch_size = batch_size

    def _build_item_vectors(self):
        # Multi-hot encode genres and tags, weight genres, L2-normalize each game vector
        # Multi-hot жанров и тегов, усиливаем жанры, L2-нормируем вектор каждой игры
        genre_lists = split_genres(self.games["Genres"])
        tag_lists = split_tags(self.games["Tags"])

        genre_mat = MultiLabelBinarizer().fit_transform(genre_lists)
        tag_mat = MultiLabelBinarizer().fit_transform(tag_lists)
        item_mat = sparse.hstack([
            sparse.csr_matrix(genre_mat) * self.genre_weight,
            sparse.csr_matrix(tag_mat),
        ]).astype(np.float32)

        self.item_ids_ = self.games[self.item_col].to_numpy()
        self.item_index_ = {g: i for i, g in enumerate(self.item_ids_)}
        # L2-normalize per item; dense for fast scoring (item count is small)
        # L2-нормируем по строкам; делаем dense для быстрого скоринга (игр немного)
        self.item_vecs_ = np.asarray(normalize(item_mat).todense(), dtype=np.float32)

    def fit(self, train: pd.DataFrame) -> "ContentBasedRecommender":
        self._build_item_vectors()
        n_items = len(self.item_ids_)

        # Keep only train rows whose game has content features
        # Оставляем только train-строки, у игр которых есть контентные признаки
        t = train[train[self.item_col].isin(self.item_index_)]
        self.user_ids_ = t[self.user_col].unique()
        self.user_index_ = {u: i for i, u in enumerate(self.user_ids_)}

        rows = t[self.user_col].map(self.user_index_).to_numpy()
        cols = t[self.item_col].map(self.item_index_).to_numpy()
        incidence = sparse.csr_matrix(
            (np.ones(len(t), dtype=np.float32), (rows, cols)),
            shape=(len(self.user_ids_), n_items),
        )

        # User profile = L2-normalized sum of L2-normalized item vectors
        # Профиль юзера = L2-нормированная сумма L2-нормированных векторов игр
        profiles = incidence @ self.item_vecs_
        self.user_profiles_ = np.asarray(normalize(profiles), dtype=np.float32)

        # Train items per user (as column indices) to mask them out at scoring
        # Train-игры каждого юзера (как индексы столбцов) для маскировки при скоринге
        self.seen_idx_ = (
            pd.Series(cols, index=t[self.user_col].to_numpy())
            .groupby(level=0).agg(list)
            .to_dict()
        )
        return self

    def recommend(self, user_ids, k: int = 10) -> dict:
        recs = {}
        user_ids = list(user_ids)
        for start in range(0, len(user_ids), self.batch_size):
            batch = user_ids[start:start + self.batch_size]
            rows = [self.user_index_[u] for u in batch if u in self.user_index_]
            valid = [u for u in batch if u in self.user_index_]
            if not valid:
                for u in batch:
                    recs[u] = []
                continue
            scores = self.user_profiles_[rows] @ self.item_vecs_.T  # (b x n_items)
            for i, u in enumerate(valid):
                row = scores[i]
                seen = self.seen_idx_.get(u)
                if seen is not None:
                    row[seen] = -np.inf
                top = np.argpartition(-row, k)[:k]
                top = top[np.argsort(-row[top])]
                recs[u] = [self.item_ids_[j] for j in top]
            for u in batch:
                if u not in self.user_index_:
                    recs[u] = []
        return recs
