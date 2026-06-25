import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from threadpoolctl import threadpool_limits

from .base import BaseRecommender


class ALSRecommender(BaseRecommender):
    # Collaborative filtering with implicit ALS on binary user-item feedback,
    # BM25-weighted (standard for implicit feedback).
    # Коллаборативная фильтрация: implicit ALS на бинарном user-item feedback
    # с BM25-взвешиванием (стандарт для implicit feedback).

    def __init__(self, factors: int = 128, iterations: int = 25,
                 regularization: float = 0.05, random_state: int = 42,
                 use_bm25: bool = True, bm25_k1: float = 100.0, bm25_b: float = 0.8,
                 user_col: str = "user_id", item_col: str = "game_id"):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.random_state = random_state
        self.use_bm25 = use_bm25
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.user_col = user_col
        self.item_col = item_col

    def fit(self, train: pd.DataFrame) -> "ALSRecommender":
        # Encode users/items to contiguous indices, build a binary user-item matrix
        # Кодируем юзеров/игры в индексы, строим бинарную user-item матрицу
        self.user_ids_ = train[self.user_col].unique()
        self.item_ids_ = train[self.item_col].unique()
        self.user_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        item_index = {it: i for i, it in enumerate(self.item_ids_)}

        rows = train[self.user_col].map(self.user_index_).to_numpy()
        cols = train[self.item_col].map(item_index).to_numpy()
        self.user_items_ = sparse.csr_matrix(
            (np.ones(len(train), dtype=np.float32), (rows, cols)),
            shape=(len(self.user_ids_), len(self.item_ids_)),
        )

        # BM25 weighting of the (binary) interactions before factorization
        # BM25-взвешивание (бинарных) взаимодействий перед факторизацией
        confidence = self.user_items_
        if self.use_bm25:
            confidence = bm25_weight(self.user_items_, K1=self.bm25_k1,
                                     B=self.bm25_b).tocsr()

        self.model_ = AlternatingLeastSquares(
            factors=self.factors, iterations=self.iterations,
            regularization=self.regularization, random_state=self.random_state,
        )
        # Single-threaded BLAS + silence the harmless MKL threadpool warning
        # Однопоточный BLAS + глушим безвредный MKL-warning о threadpool
        with threadpool_limits(1, "blas"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(confidence, show_progress=False)
        return self

    def recommend(self, user_ids, k: int = 10) -> dict:
        user_ids = list(user_ids)
        rows = np.array([self.user_index_[u] for u in user_ids])
        ids, _ = self.model_.recommend(
            rows, self.user_items_[rows], N=k,
            filter_already_liked_items=True,
        )
        return {u: [self.item_ids_[j] for j in ids[i]] for i, u in enumerate(user_ids)}
