import numpy as np

from ..baselines.base import BaseRecommender
from .candidates import fuse_candidates
from .features import build_features


class RerankRecommender(BaseRecommender):
    # Two-stage recommender: multi-source retrieval generates candidates,
    # a LambdaMART model re-ranks them with rich user-item features.
    # Двухстадийный рекомендер: multi-source retrieval даёт кандидатов,
    # модель LambdaMART переупорядочивает их по богатым признакам.

    def __init__(self, retrievers: dict, ctx: dict, model, k_per: int = 30):
        self.retrievers = retrievers
        self.ctx = ctx
        self.model = model
        self.k_per = k_per

    def fit(self, train=None) -> "RerankRecommender":
        # Retrievers and the LambdaMART model are already fitted upstream
        # Ретриверы и модель LambdaMART уже обучены ранее
        return self

    def recommend(self, user_ids, k: int = 10) -> dict:
        user_index = self.ctx["user_index"]
        item_index = self.ctx["item_index"]
        item_ids = self.ctx["item_ids"]

        fused = fuse_candidates(self.retrievers, user_ids, k_per=self.k_per)

        # Flatten all (user, candidate) pairs, keeping per-user slices
        # Разворачиваем все пары (user, candidate), запоминая срезы по юзерам
        u_rows, i_rows, owners, cand_items = [], [], [], []
        slices = {}
        for u in user_ids:
            if u not in user_index:
                continue
            cands = [g for g in fused.get(u, []) if g in item_index]
            if not cands:
                continue
            start = len(i_rows)
            for g in cands:
                u_rows.append(user_index[u])
                i_rows.append(item_index[g])
                cand_items.append(g)
            slices[u] = (start, len(i_rows))

        recs = {u: [] for u in user_ids}
        if not i_rows:
            return recs

        X = build_features(self.ctx, np.array(u_rows), np.array(i_rows))
        scores = self.model.predict(X)
        cand_items = np.array(cand_items)

        for u, (a, b) in slices.items():
            s = scores[a:b]
            order = np.argsort(-s)[:k]
            recs[u] = [cand_items[a + j] for j in order]
        return recs
