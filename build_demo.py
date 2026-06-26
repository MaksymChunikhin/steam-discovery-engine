"""Auto-generate demo data so anyone can try the API without hunting for IDs.

Picks representative real users per dominant genre (FPS / RPG / Strategy / ...)
and a set of recognizable featured games. Output: data/demo/.

Авто-генерация демо-данных, чтобы любой мог попробовать API без поиска ID.
Выбирает репрезентативных реальных юзеров по доминирующему жанру и набор
узнаваемых игр. Результат: data/demo/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

# Dominant primary genre -> demo archetype display name
# Доминирующий основной жанр -> отображаемое имя архетипа
ARCHETYPES = {
    "Action": "Action Gamer",
    "RPG": "RPG Explorer",
    "Strategy": "Strategy Player",
    "Simulation": "Simulation Fan",
    "Racing": "Racing Fan",
    "Indie": "Indie Lover",
    "Adventure": "Adventure Seeker",
    "Sports": "Sports Fan",
}

# Recognizable titles to feature if present in our catalog
# Узнаваемые тайтлы для витрины, если есть в каталоге
WANTED_GAMES = [
    "Counter-Strike 2", "Cyberpunk 2077", "Stardew Valley", "Terraria",
    "Hades", "Factorio", "DOOM", "Dead Cells", "Euro Truck Simulator 2",
    "Sid Meier's Civilization VI",
]


def main():
    train = pd.read_parquet(DATA / "processed" / "train.parquet")
    games = pd.read_parquet(DATA / "processed" / "games.parquet")

    primary = (games.set_index("game_id")["Genres"].fillna("")
               .map(lambda s: s.split(",")[0].strip() if s else "")).to_dict()

    t = train[train["game_id"].isin(primary)].copy()
    t["genre"] = t["game_id"].map(primary)
    t["hours"] = t["playtime_forever"] / 60.0

    # Per-user profile: dominant genre, its share, number of games, avg playtime
    # Профиль юзера: доминирующий жанр, его доля, число игр, среднее время игры
    per_user = t.groupby("user_id").agg(games=("game_id", "size"),
                                        avg_playtime=("hours", "mean"))
    dom = (t.groupby(["user_id", "genre"]).size().rename("cnt").reset_index())
    dom["share"] = dom["cnt"] / dom.groupby("user_id")["cnt"].transform("sum")
    dom = dom.sort_values(["user_id", "cnt"]).groupby("user_id").tail(1).set_index("user_id")
    prof = per_user.join(dom[["genre", "share"]])
    prof = prof[prof["games"] >= 8]  # enough history to be characteristic

    rows = []
    for genre, name in ARCHETYPES.items():
        cand = prof[prof["genre"] == genre]
        if not len(cand):
            continue
        # Most characteristic: highest genre share, tie-break by number of games
        # Самый характерный: макс. доля жанра, тай-брейк по числу игр
        pick = cand.sort_values(["share", "games"], ascending=False).iloc[0]
        rows.append({
            "display_name": name,
            "user_id": int(pick.name),
            "favorite_genre": genre,
            "games": int(pick["games"]),
            "avg_playtime": round(float(pick["avg_playtime"]), 1),
        })

    demo_users = pd.DataFrame(rows)

    # Featured games: wanted titles present in catalog, padded with most popular
    # Витрина игр: нужные тайтлы из каталога, добитые самыми популярными
    pop = train["game_id"].value_counts()
    g = games.set_index("game_id")
    featured = []
    seen = set()
    for title in WANTED_GAMES:
        m = games[games["Name"] == title]
        if len(m):
            gid = int(m["game_id"].iloc[0])
            if gid not in seen:
                seen.add(gid)
                featured.append(gid)
    for gid in pop.index:
        if len(featured) >= 10:
            break
        if int(gid) not in seen and gid in g.index:
            seen.add(int(gid))
            featured.append(int(gid))
    featured_games = pd.DataFrame([
        {"game_id": gid, "title": g.loc[gid, "Name"], "genres": str(g.loc[gid, "Genres"])}
        for gid in featured
    ])

    out = DATA / "demo"
    out.mkdir(parents=True, exist_ok=True)
    demo_users.to_csv(out / "demo_users.csv", index=False)
    featured_games.to_csv(out / "featured_games.csv", index=False)
    print("demo users:\n", demo_users.to_string(index=False))
    print("\nfeatured games:\n", featured_games.to_string(index=False))
    print("\nsaved to", out)


if __name__ == "__main__":
    main()
