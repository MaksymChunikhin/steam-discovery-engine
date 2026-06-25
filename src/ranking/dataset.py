import numpy as np
import pandas as pd

from .candidates import fuse_candidates
from .features import build_features


def build_training_dataset(rank_holdout, retrievers, ctx, k_per: int = 30,
                           n_users: int = None, seed: int = 42,
                           user_col: str = "user_id", item_col: str = "game_id"):
    # Build the LambdaMART training set from a held-out interaction per user that
    # the retrievers did NOT see (retrievers are fit on train_fit). This matches
    # the test regime, where the target is also unseen — avoiding score leakage.
    # The held-out item is the positive; fused retrieved games are the negatives.
    # Query group = user.
    # Тренировочный набор LambdaMART из held-out взаимодействия на юзера, которое
    # ретриверы НЕ видели (они обучены на train_fit). Это совпадает с тестовым режимом,
    # где target тоже не виден, и устраняет утечку скоров. Held-out игра — позитив,
    # найденные ретриверами игры — негативы. Query-группа = user.
    item_index = ctx["item_index"]
    user_index = ctx["user_index"]

    target = (rank_holdout[rank_holdout[item_col].isin(item_index)
                           & rank_holdout[user_col].isin(user_index)]
              .groupby(user_col)[item_col].last().to_dict())
    users = list(target.keys())
    if n_users is not None and n_users < len(users):
        rng = np.random.default_rng(seed)
        users = list(rng.choice(users, size=n_users, replace=False))
    fused = fuse_candidates(retrievers, users, k_per=k_per)

    user_rows, item_rows, labels, groups = [], [], [], []
    for u in users:
        tgt = target[u]
        negs = [g for g in fused.get(u, []) if g != tgt and g in item_index]
        if not negs:
            continue
        ur = user_index[u]
        user_rows.append(ur)
        item_rows.append(item_index[tgt])
        labels.append(1)
        for g in negs:
            user_rows.append(ur)
            item_rows.append(item_index[g])
            labels.append(0)
        groups.append(1 + len(negs))

    user_rows = np.array(user_rows)
    item_rows = np.array(item_rows)
    X = build_features(ctx, user_rows, item_rows)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)
    return X, y, groups
