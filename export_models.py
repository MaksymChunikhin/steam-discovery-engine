"""Offline training -> export artifacts for the inference service.

Fits the retrieval models, trains the leakage-free LambdaMART re-ranker, and
saves everything the FastAPI service needs into models/. Serving then loads
these artifacts and never trains.

Оффлайн-обучение -> экспорт артефактов для инференс-сервиса.
Обучает retrieval-модели, тренирует re-ranker LambdaMART без утечки и сохраняет
всё, что нужно сервису, в models/. Сервинг затем только загружает артефакты.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
MODELS = ROOT / "models"

from src.baselines import (PopularityRecommender, ALSRecommender,
                           SemanticEmbeddingRecommender, TwoTowerRecommender)
from src.evaluation import leave_last_out
from src.ranking import build_context, build_training_dataset, train_reranker, FEATURE_NAMES

K_PER = 50
TT_EPOCHS = 10


def fit_retrievers(tr, games, game_emb, game_ids):
    return {
        "als": ALSRecommender(factors=128, iterations=25, regularization=0.05).fit(tr),
        "two_tower": TwoTowerRecommender(games, game_emb, game_ids, epochs=TT_EPOCHS, seed=42).fit(tr),
        "semantic": SemanticEmbeddingRecommender(game_emb, game_ids).fit(tr),
        "popularity": PopularityRecommender().fit(tr),
    }


def main():
    train = pd.read_parquet(DATA / "processed" / "train.parquet")
    games = pd.read_parquet(DATA / "processed" / "games.parquet")
    emb_meta = pd.read_parquet(DATA / "processed" / "game_embeddings_metadata.parquet").sort_values("embedding_idx")
    game_emb = np.load(DATA / "processed" / "game_embeddings.npy")
    game_ids = emb_meta["game_id"].to_numpy()

    # Leakage-free re-ranker training (retrievers on train_fit, targets held out)
    print("[1/4] fitting retrievers on train_fit for leakage-free ranking dataset...")
    train_fit, rank_holdout = leave_last_out(train)
    retr_fit = fit_retrievers(train_fit, games, game_emb, game_ids)
    ctx_fit = build_context(train_fit, games, game_ids, game_emb, retr_fit["als"], retr_fit["two_tower"])
    X, y, groups = build_training_dataset(rank_holdout, retr_fit, ctx_fit, k_per=K_PER)
    print(f"      ranking dataset: {X.shape}, positives: {int(y.sum())}")
    rr_model = train_reranker(X, y, groups)

    # Final models on the full train set
    print("[2/4] fitting retrievers on full train...")
    retr_full = fit_retrievers(train, games, game_emb, game_ids)
    ctx = build_context(train, games, game_ids, game_emb, retr_full["als"], retr_full["two_tower"])

    print("[3/4] saving artifacts to models/ ...")
    for sub in ["als", "semantic", "two_tower", "ranking", "content", "metadata"]:
        (MODELS / sub).mkdir(parents=True, exist_ok=True)

    np.savez(MODELS / "als" / "factors.npz", user_factors=ctx["als_uf"], item_factors=ctx["als_if"])
    np.savez(MODELS / "semantic" / "embeddings.npz", user_sem=ctx["user_sem"], item_sem=ctx["item_sem"])
    np.savez(MODELS / "two_tower" / "embeddings.npz", user_tt=ctx["user_tt"], item_tt=ctx["item_tt"])
    torch.save(retr_full["two_tower"].model_.state_dict(), MODELS / "two_tower" / "model.pt")
    np.savez(MODELS / "content" / "features.npz",
             G=ctx["G"], T=ctx["T"], genre_prof=ctx["genre_prof"], tag_prof=ctx["tag_prof"],
             price=ctx["price"], review_score=ctx["review_score"],
             popularity=ctx["popularity"], pop_rank=ctx["pop_rank"], user_num=ctx["user_num"])
    rr_model.booster_.save_model(str(MODELS / "ranking" / "lambdamart.txt"))

    np.save(MODELS / "metadata" / "game_ids.npy", ctx["item_ids"])
    np.save(MODELS / "metadata" / "user_ids.npy", ctx["user_ids"])
    with open(MODELS / "metadata" / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    # Seen items per user (to exclude already-played games at serving time)
    seen = train[train["game_id"].isin(set(ctx["item_index"]))][["user_id", "game_id"]]
    seen.to_parquet(MODELS / "metadata" / "seen.parquet", index=False)

    config = {
        "k_per": K_PER,
        "n_users": int(len(ctx["user_ids"])),
        "n_items": int(len(ctx["item_ids"])),
        "tt_dim": int(ctx["item_tt"].shape[1]),
        "semantic_dim": int(ctx["item_sem"].shape[1]),
        "feature_names": FEATURE_NAMES,
        "exported": str(date.today()),
    }
    with open(MODELS / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("[4/4] done. artifacts in", MODELS)


if __name__ == "__main__":
    main()
