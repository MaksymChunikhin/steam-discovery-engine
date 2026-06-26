import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import faiss

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"

# Dominant retrieval signal -> human-readable reason
# Доминирующий retrieval-сигнал -> понятная причина
REASONS = {
    "als_score": "Players with similar gaming histories also enjoyed this title.",
    "semantic_score": "Similar gameplay and review semantics to games you played.",
    "two_tower_score": "Matches your learned taste profile.",
    "popularity_rank": "Popular among the Steam community.",
}
_REASON_FEATURES = list(REASONS)


def _flat_index(matrix):
    # Cosine via inner product on L2-normalized vectors
    # Cosine через inner product на L2-нормированных векторах
    emb = np.ascontiguousarray(matrix, dtype=np.float32)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


class RecommendationPipeline:
    # Self-contained inference: loads exported artifacts (numpy / lightgbm / faiss only)
    # and serves the two-stage pipeline behind .recommend() / .similar().
    # Самодостаточный инференс: грузит экспортированные артефакты (только numpy / lightgbm /
    # faiss) и отдаёт двухстадийный пайплайн за .recommend() / .similar().

    def __init__(self, ctx, booster, feature_names, games, demo, k_per):
        self.ctx = ctx
        self.booster = booster
        self.feature_names = feature_names
        self.games = games
        self.demo = demo
        self.k_per = k_per
        self.tt_index = _flat_index(ctx["item_tt"].copy())
        self.sem_index = _flat_index(ctx["item_sem"].copy())
        self._genre_by_id = dict(zip(games["game_id"], games["Genres"].fillna("")))

    @classmethod
    def load(cls, models_dir: Path = MODELS):
        config = json.loads((models_dir / "config.json").read_text())
        als = np.load(models_dir / "als" / "factors.npz")
        sem = np.load(models_dir / "semantic" / "embeddings.npz")
        tt = np.load(models_dir / "two_tower" / "embeddings.npz")
        content = np.load(models_dir / "content" / "features.npz")
        item_ids = np.load(models_dir / "metadata" / "game_ids.npy", allow_pickle=True)
        user_ids = np.load(models_dir / "metadata" / "user_ids.npy", allow_pickle=True)
        feature_names = json.loads((models_dir / "metadata" / "feature_names.json").read_text())

        ctx = {
            "item_ids": item_ids,
            "item_index": {int(g): i for i, g in enumerate(item_ids)},
            "user_ids": user_ids,
            "user_index": {int(u): i for i, u in enumerate(user_ids)},
            "als_uf": als["user_factors"], "als_if": als["item_factors"],
            "user_sem": sem["user_sem"], "item_sem": sem["item_sem"],
            "user_tt": tt["user_tt"], "item_tt": tt["item_tt"],
            "G": content["G"], "T": content["T"],
            "genre_prof": content["genre_prof"], "tag_prof": content["tag_prof"],
            "price": content["price"], "review_score": content["review_score"],
            "popularity": content["popularity"], "pop_rank": content["pop_rank"],
            "user_num": content["user_num"],
        }

        seen_df = pd.read_parquet(models_dir / "metadata" / "seen.parquet")
        seen_df = seen_df[seen_df["game_id"].isin(ctx["item_index"])]
        seen_df["ui"] = seen_df["user_id"].map(ctx["user_index"])
        seen_df["it"] = seen_df["game_id"].map(ctx["item_index"])
        ctx["seen"] = {int(u): np.asarray(v, dtype=np.int64)
                       for u, v in seen_df.groupby("ui")["it"].agg(list).items()}

        booster = lgb.Booster(model_file=str(models_dir / "ranking" / "lambdamart.txt"))
        games = pd.read_parquet(DATA / "processed" / "games.parquet")

        demo = {"users": [], "games": []}
        demo_dir = DATA / "demo"
        if (demo_dir / "demo_users.csv").exists():
            demo["users"] = pd.read_csv(demo_dir / "demo_users.csv").to_dict("records")
        if (demo_dir / "featured_games.csv").exists():
            demo["games"] = pd.read_csv(demo_dir / "featured_games.csv").to_dict("records")

        return cls(ctx, booster, feature_names, games, demo, config["k_per"])

    # --- features (must mirror src/ranking/features.py) ---
    # --- фичи (должны зеркалить src/ranking/features.py) ---

    def _features(self, u, item_rows):
        c = self.ctx
        un = c["user_num"][u]
        cols = {
            "als_score": c["als_if"][item_rows] @ c["als_uf"][u],
            "semantic_score": c["item_sem"][item_rows] @ c["user_sem"][u],
            "two_tower_score": c["item_tt"][item_rows] @ c["user_tt"][u],
            "popularity_rank": c["pop_rank"][item_rows],
            "genre_overlap": c["G"][item_rows] @ c["genre_prof"][u],
            "tag_overlap": c["T"][item_rows] @ c["tag_prof"][u],
            "price_difference": np.abs(c["price"][item_rows] - un[2]),
            "review_score": c["review_score"][item_rows],
            "game_popularity": np.log1p(c["popularity"][item_rows]),
            "price": c["price"][item_rows],
            "avg_playtime": np.full(len(item_rows), un[0], dtype=np.float32),
            "positive_ratio": np.full(len(item_rows), un[1], dtype=np.float32),
            "n_games": np.full(len(item_rows), un[3], dtype=np.float32),
        }
        return pd.DataFrame({name: cols[name] for name in self.feature_names})

    # --- candidate fusion ---

    def _topk(self, scores, seen, k):
        scores = scores.copy()
        if len(seen):
            scores[seen] = -np.inf
        idx = np.argpartition(-scores, k)[:k]
        return idx[np.argsort(-scores[idx])]

    def _fuse(self, u):
        c, k = self.ctx, self.k_per
        seen = c["seen"].get(u, np.empty(0, dtype=np.int64))
        seen_set = set(seen.tolist())
        als = self._topk(c["als_uf"][u] @ c["als_if"].T, seen, k)
        semc = self._topk(c["user_sem"][u] @ c["item_sem"].T, seen, k)
        popc = self._topk(c["pop_rank"].astype(np.float32), seen, k)
        _, tt = self.tt_index.search(
            np.ascontiguousarray(c["user_tt"][u:u + 1], dtype=np.float32), k + len(seen) + 1)
        tt = [j for j in tt[0] if j not in seen_set][:k]
        out, taken = [], set()
        for j in list(als) + list(tt) + list(semc) + list(popc):
            j = int(j)
            if j not in taken:
                taken.add(j)
                out.append(j)
        return out

    def _reason(self, feats, j, stats):
        best, best_z = _REASON_FEATURES[0], -np.inf
        for f in _REASON_FEATURES:
            mu, sd = stats[f]
            z = (feats[f].iloc[j] - mu) / sd if sd > 0 else 0.0
            if z > best_z:
                best_z, best = z, f
        return REASONS[best]

    # --- public API ---

    def has_user(self, user_id):
        return int(user_id) in self.ctx["user_index"]

    def recommend(self, user_id, k=10):
        c = self.ctx
        u = c["user_index"][int(user_id)]
        cands = self._fuse(u)
        feats = self._features(u, np.array(cands))
        scores = self.booster.predict(feats)
        order = np.argsort(-scores)[:k]
        stats = {f: (feats[f].mean(), feats[f].std()) for f in _REASON_FEATURES}
        out = []
        for j in order:
            gid = int(c["item_ids"][cands[j]])
            out.append({"game_id": gid, "title": self._title(gid),
                        "score": round(float(scores[j]), 4),
                        "reason": self._reason(feats, j, stats),
                        "genres": str(self._genre_by_id.get(gid, ""))})
        return out

    def similar(self, game_id, k=10):
        c = self.ctx
        if int(game_id) not in c["item_index"]:
            return None
        row = c["item_index"][int(game_id)]
        _, idx = self.sem_index.search(
            np.ascontiguousarray(c["item_sem"][row:row + 1], dtype=np.float32), k + 1)
        out = []
        for j in idx[0]:
            j = int(j)
            if j == row:
                continue
            gid = int(c["item_ids"][j])
            out.append({"game_id": gid, "title": self._title(gid)})
            if len(out) == k:
                break
        return out

    def game_info(self, game_id):
        rows = self.games[self.games["game_id"] == int(game_id)]
        if not len(rows):
            return None
        r = rows.iloc[0]
        return {"game_id": int(game_id), "title": r["Name"],
                "genres": str(r["Genres"]), "tags": str(r["Tags"]),
                "price": float(r["Price"]) if pd.notna(r["Price"]) else 0.0}

    def _title(self, gid):
        r = self.games[self.games["game_id"] == gid]
        return r["Name"].iloc[0] if len(r) else str(gid)
