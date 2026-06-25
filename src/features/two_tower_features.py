import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from .game_features import split_genres, split_tags


def build_two_tower_data(train: pd.DataFrame, games: pd.DataFrame,
                         game_emb: np.ndarray, game_ids,
                         user_col: str = "user_id", item_col: str = "game_id"):
    # Build item and user feature tensors for the Two-Tower model.
    # Item: review embedding + multi-hot genres/tags + price + review score.
    # User: genre/tag history profile + numeric stats (all from train only).
    # Признаки айтемов и пользователей для Two-Tower.
    # Item: эмбеддинг отзывов + multi-hot жанры/теги + цена + review score.
    # User: профиль жанров/тегов по истории + числовые статистики (только из train).
    item_ids = np.asarray(game_ids)
    item_index = {g: i for i, g in enumerate(item_ids)}

    gsub = games.set_index(item_col).reindex(item_ids)

    G = MultiLabelBinarizer().fit_transform(split_genres(gsub["Genres"])).astype(np.float32)
    T = MultiLabelBinarizer().fit_transform(split_tags(gsub["Tags"])).astype(np.float32)
    price = gsub["Price"].fillna(0).to_numpy(dtype=np.float64)
    pos = gsub["Positive"].fillna(0).to_numpy(dtype=np.float64)
    neg = gsub["Negative"].fillna(0).to_numpy(dtype=np.float64)
    review_score = pos / np.maximum(pos + neg, 1.0)

    price_scaled = StandardScaler().fit_transform(price.reshape(-1, 1)).ravel()
    item_struct = np.hstack([
        G, T, price_scaled.reshape(-1, 1), review_score.reshape(-1, 1)
    ]).astype(np.float32)
    item_emb = np.ascontiguousarray(game_emb, dtype=np.float32)

    # Train interactions restricted to items that have features
    # Train-взаимодействия, ограниченные айтемами с признаками
    t = train[train[item_col].isin(item_index)].copy()
    t["it"] = t[item_col].map(item_index).to_numpy()
    user_ids = t[user_col].unique()
    user_index = {u: i for i, u in enumerate(user_ids)}
    t["ui"] = t[user_col].map(user_index).to_numpy()
    n_users, n_items = len(user_ids), len(item_ids)

    # History profiles (normalized genre/tag counts over the user's games)
    # Профили истории (нормированные счётчики жанров/тегов по играм юзера)
    R = sparse.csr_matrix((np.ones(len(t), np.float32), (t["ui"], t["it"])),
                          shape=(n_users, n_items))
    genre_prof = R @ G
    tag_prof = R @ T
    genre_prof /= np.maximum(genre_prof.sum(1, keepdims=True), 1.0)
    tag_prof /= np.maximum(tag_prof.sum(1, keepdims=True), 1.0)
    user_hist = np.hstack([genre_prof, tag_prof]).astype(np.float32)

    # Numeric user stats
    # Числовые статистики пользователя
    t["hours"] = np.log1p(t["playtime_forever"].to_numpy(dtype=np.float64) / 60.0)
    t["price"] = price[t["it"].to_numpy()]
    agg = t.groupby("ui").agg(avg_hours=("hours", "mean"),
                              pos_ratio=("label", "mean"),
                              avg_price=("price", "mean"),
                              n_games=("it", "size")).sort_index()
    num = agg[["avg_hours", "pos_ratio", "avg_price", "n_games"]].to_numpy(dtype=np.float64)
    num[:, [0, 2, 3]] = StandardScaler().fit_transform(num[:, [0, 2, 3]])  # keep pos_ratio raw
    user_num = num.astype(np.float32)

    pairs = np.stack([t["ui"].to_numpy(), t["it"].to_numpy()], axis=1).astype(np.int64)
    seen = t.groupby("ui")["it"].agg(set).to_dict()

    return {
        "item_ids": item_ids, "item_index": item_index,
        "user_ids": user_ids, "user_index": user_index,
        "item_emb": item_emb, "item_struct": item_struct,
        "user_hist": user_hist, "user_num": user_num,
        "pairs": pairs, "seen": seen,
    }
