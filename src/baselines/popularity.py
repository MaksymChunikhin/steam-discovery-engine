import pandas as pd

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    # Recommend the globally most popular games, skipping items the user already has.
    # Рекомендует глобально самые популярные игры, пропуская уже виденные.

    def __init__(self, user_col: str = "user_id", item_col: str = "game_id"):
        self.user_col = user_col
        self.item_col = item_col

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        # Most popular games by interaction count, and each user's seen items
        # Самые популярные игры по числу взаимодействий и виденные игры каждого юзера
        self.popular_ = train[self.item_col].value_counts().index.tolist()
        self.seen_ = train.groupby(self.user_col)[self.item_col].agg(set).to_dict()
        return self

    def recommend(self, user_ids, k: int = 10) -> dict:
        recs = {}
        for u in user_ids:
            seen = self.seen_.get(u, set())
            out = []
            for g in self.popular_:
                if g not in seen:
                    out.append(g)
                    if len(out) == k:
                        break
            recs[u] = out
        return recs
