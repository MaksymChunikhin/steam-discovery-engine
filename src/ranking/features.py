import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from ..features.game_features import split_genres, split_tags


def build_context(train, games, game_ids, game_emb, als, two_tower,
                  user_col="user_id", item_col="game_id"):
    # Precompute every array the ranking features need, aligned to a single
    # user order (the Two-Tower order) and item order (game_ids).
    # Предрасчёт всех массивов для ранжирующих фич, выровненных по единому порядку
    # пользователей (порядок Two-Tower) и игр (game_ids).
    item_ids = np.asarray(game_ids)
    item_index = {g: i for i, g in enumerate(item_ids)}
    user_ids = two_tower.data_["user_ids"]
    user_index = {u: i for i, u in enumerate(user_ids)}
    nU, nI = len(user_ids), len(item_ids)

    gsub = games.set_index(item_col).reindex(item_ids)
    G = MultiLabelBinarizer().fit_transform(split_genres(gsub["Genres"])).astype(np.float32)
    T = MultiLabelBinarizer().fit_transform(split_tags(gsub["Tags"])).astype(np.float32)
    price = gsub["Price"].fillna(0).to_numpy(np.float32)
    pos = gsub["Positive"].fillna(0).to_numpy(np.float64)
    neg = gsub["Negative"].fillna(0).to_numpy(np.float64)
    review_score = (pos / np.maximum(pos + neg, 1.0)).astype(np.float32)

    # Train interactions restricted to items with features
    # Train-взаимодействия, ограниченные айтемами с признаками
    t = train[train[item_col].isin(item_index)].copy()
    t["ui"] = t[user_col].map(user_index).to_numpy()
    t["it"] = t[item_col].map(item_index).to_numpy()
    t = t.dropna(subset=["ui"])
    t["ui"] = t["ui"].astype(np.int64)

    R = sparse.csr_matrix((np.ones(len(t), np.float32), (t["ui"], t["it"])), shape=(nU, nI))
    popularity = np.asarray(R.sum(0)).ravel().astype(np.float32)          # per-item count
    pop_rank = popularity.argsort().argsort().astype(np.float32) / nI     # 0..1 normalized rank

    genre_prof = R @ G
    genre_prof /= np.maximum(genre_prof.sum(1, keepdims=True), 1.0)
    tag_prof = R @ T
    tag_prof /= np.maximum(tag_prof.sum(1, keepdims=True), 1.0)

    item_sem = np.ascontiguousarray(game_emb, dtype=np.float32)
    user_sem = np.asarray(normalize(R @ item_sem), dtype=np.float32)

    # Learned Two-Tower embeddings (already aligned to game_ids / user order)
    # Обученные эмбеддинги Two-Tower (уже выровнены по game_ids / порядку юзеров)
    item_tt = np.ascontiguousarray(two_tower.item_emb_out_, dtype=np.float32)
    user_tt = np.ascontiguousarray(two_tower._user_embeddings(np.arange(nU)), dtype=np.float32)

    # ALS latent factors aligned to our indices
    # Латентные факторы ALS, выровненные по нашим индексам
    als_u_pos = [als.user_index_[u] for u in user_ids]
    als_item_pos = {it: i for i, it in enumerate(als.item_ids_)}
    als_i_pos = [als_item_pos[g] for g in item_ids]
    als_uf = np.ascontiguousarray(als.model_.user_factors[als_u_pos], dtype=np.float32)
    als_if = np.ascontiguousarray(als.model_.item_factors[als_i_pos], dtype=np.float32)

    # Numeric user stats: avg log1p playtime hours, positive ratio, avg price, n_games
    # Числовые статы юзера: среднее log1p часов, доля позитивных, средняя цена, число игр
    t["hours"] = np.log1p(t["playtime_forever"].to_numpy(np.float64) / 60.0)
    t["price"] = price[t["it"].to_numpy()]
    agg = t.groupby("ui").agg(avg_hours=("hours", "mean"),
                              pos_ratio=("label", "mean"),
                              avg_price=("price", "mean"),
                              n_games=("it", "size")).sort_index()
    user_num = agg.reindex(range(nU)).fillna(0).to_numpy(np.float32)  # cols: hours,pos,price,n

    return {
        "item_ids": item_ids, "item_index": item_index,
        "user_ids": user_ids, "user_index": user_index,
        "G": G, "T": T, "price": price, "review_score": review_score,
        "popularity": popularity, "pop_rank": pop_rank,
        "genre_prof": genre_prof, "tag_prof": tag_prof,
        "item_sem": item_sem, "user_sem": user_sem,
        "item_tt": item_tt, "user_tt": user_tt,
        "als_uf": als_uf, "als_if": als_if,
        "user_num": user_num,
    }


def _rowdot(A, urows, B, irows):
    # Row-wise dot product of A[urows] and B[irows]
    # Построчное скалярное произведение A[urows] и B[irows]
    return np.einsum("ij,ij->i", A[urows], B[irows]).astype(np.float32)


# Feature registry: grouped by source. Adding/removing a feature is one line.
# Реестр фич, сгруппированный по источнику. Добавить/убрать фичу — одна строка.
FEATURE_BUILDERS = [
    # Retrieval features
    ("als_score", lambda c, u, i: _rowdot(c["als_uf"], u, c["als_if"], i)),
    ("semantic_score", lambda c, u, i: _rowdot(c["user_sem"], u, c["item_sem"], i)),
    ("two_tower_score", lambda c, u, i: _rowdot(c["user_tt"], u, c["item_tt"], i)),
    ("popularity_rank", lambda c, u, i: c["pop_rank"][i]),
    # Content features
    ("genre_overlap", lambda c, u, i: _rowdot(c["genre_prof"], u, c["G"], i)),
    ("tag_overlap", lambda c, u, i: _rowdot(c["tag_prof"], u, c["T"], i)),
    ("price_difference", lambda c, u, i: np.abs(c["price"][i] - c["user_num"][u, 2])),
    # Game features
    ("review_score", lambda c, u, i: c["review_score"][i]),
    ("game_popularity", lambda c, u, i: np.log1p(c["popularity"][i])),
    ("price", lambda c, u, i: c["price"][i]),
    # User features
    ("avg_playtime", lambda c, u, i: c["user_num"][u, 0]),
    ("positive_ratio", lambda c, u, i: c["user_num"][u, 1]),
    ("n_games", lambda c, u, i: c["user_num"][u, 3]),
]

FEATURE_NAMES = [name for name, _ in FEATURE_BUILDERS]


def build_features(ctx, user_rows, item_rows, chunk_size=500_000):
    # Build the full feature matrix for candidate (user, item) pairs, in chunks
    # Строим полную матрицу фич для пар-кандидатов (user, item), по чанкам
    user_rows = np.asarray(user_rows)
    item_rows = np.asarray(item_rows)
    n = len(user_rows)
    cols = {name: np.empty(n, dtype=np.float32) for name in FEATURE_NAMES}
    for start in range(0, n, chunk_size):
        u = user_rows[start:start + chunk_size]
        i = item_rows[start:start + chunk_size]
        for name, fn in FEATURE_BUILDERS:
            cols[name][start:start + chunk_size] = fn(ctx, u, i)
    return pd.DataFrame(cols)
