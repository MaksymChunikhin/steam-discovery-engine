import numpy as np

from . import metrics


def evaluate(model, train, test, k: int = 10,
             user_col: str = "user_id", item_col: str = "game_id") -> dict:
    # Evaluate any BaseRecommender against the leave-last-out test set.
    # Ground truth is the set of test items per user (one item under leave-last-out).
    # Оцениваем любой BaseRecommender на leave-last-out test.
    # Ground truth — множество test-игр на юзера (одна игра при leave-last-out).
    truth = test.groupby(user_col)[item_col].agg(set).to_dict()
    users = list(truth.keys())
    recs = model.recommend(users, k=k)

    recalls, maps, ndcgs = [], [], []
    for u in users:
        relevant = truth[u]
        recommended = recs.get(u, [])
        recalls.append(metrics.recall_at_k(recommended, relevant, k))
        maps.append(metrics.average_precision_at_k(recommended, relevant, k))
        ndcgs.append(metrics.ndcg_at_k(recommended, relevant, k))

    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"MAP@{k}": float(np.mean(maps)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
    }
