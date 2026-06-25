import numpy as np
import pandas as pd
import torch

from .base import BaseRecommender
from ..features.two_tower_features import build_two_tower_data
from ..retrieval.two_tower import train_two_tower
from ..retrieval.faiss_index import build_index, search


class TwoTowerRecommender(BaseRecommender):
    # Neural retrieval: jointly learned user and item towers, FAISS for candidate search.
    # Нейронный retrieval: совместно обучаемые башни юзера и айтема, FAISS для поиска кандидатов.

    def __init__(self, games, game_emb, game_ids, epochs: int = 10,
                 batch_size: int = 1024, lr: float = 1e-3, out_dim: int = 64,
                 temperature: float = 0.05, dropout: float = 0.2,
                 device: str = None, seed: int = 42,
                 user_col: str = "user_id", item_col: str = "game_id"):
        self.games = games
        self.game_emb = game_emb
        self.game_ids = game_ids
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.out_dim = out_dim
        self.temperature = temperature
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.user_col = user_col
        self.item_col = item_col

    def fit(self, train: pd.DataFrame, eval_test: pd.DataFrame = None) -> "TwoTowerRecommender":
        self.data_ = build_two_tower_data(train, self.games, self.game_emb,
                                          self.game_ids, self.user_col, self.item_col)
        d = self.data_

        # Optional: build monitoring arrays from test (measured only, never trained on)
        # Опционально: массивы для мониторинга из test (только измеряем, не обучаемся)
        eval_data = None
        if eval_test is not None:
            et = eval_test[eval_test[self.user_col].isin(d["user_index"])
                           & eval_test[self.item_col].isin(d["item_index"])]
            eval_rows = et[self.user_col].map(d["user_index"]).to_numpy()
            eval_truth = et[self.item_col].map(d["item_index"]).to_numpy()
            eval_seen = [list(d["seen"].get(int(u), ())) for u in eval_rows]
            eval_data = (eval_rows, eval_truth, eval_seen)

        self.model_, self.history_ = train_two_tower(
            d, epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
            out_dim=self.out_dim, temperature=self.temperature, dropout=self.dropout,
            device=self.device, eval_data=eval_data, seed=self.seed)

        d = self.data_
        self.model_.eval()
        with torch.no_grad():
            item_emb = torch.tensor(d["item_emb"], device=self.device)
            item_struct = torch.tensor(d["item_struct"], device=self.device)
            self.item_emb_out_ = self.model_.item_tower(item_emb, item_struct).cpu().numpy()

        # FAISS index over learned item embeddings
        # FAISS индекс по обученным эмбеддингам айтемов
        self.index_ = build_index(self.item_emb_out_)
        self.item_ids_ = d["item_ids"]
        self.user_index_ = d["user_index"]
        self.seen_ = d["seen"]
        self._max_seen = max((len(s) for s in self.seen_.values()), default=0)
        return self

    def _user_embeddings(self, rows):
        d = self.data_
        with torch.no_grad():
            hist = torch.tensor(d["user_hist"][rows], device=self.device)
            num = torch.tensor(d["user_num"][rows], device=self.device)
            return self.model_.user_tower(hist, num).cpu().numpy()

    def recommend(self, user_ids, k: int = 10) -> dict:
        user_ids = list(user_ids)
        valid = [u for u in user_ids if u in self.user_index_]
        recs = {u: [] for u in user_ids if u not in self.user_index_}
        if not valid:
            return recs

        rows = [self.user_index_[u] for u in valid]
        u_emb = self._user_embeddings(rows)
        n_take = min(k + self._max_seen + 1, len(self.item_ids_))
        _, idx = search(self.index_, u_emb, n_take)

        for r, u in enumerate(valid):
            seen = self.seen_.get(self.user_index_[u], set())
            out = []
            for j in idx[r]:
                if j not in seen:
                    out.append(self.item_ids_[j])
                    if len(out) == k:
                        break
            recs[u] = out
        return recs
