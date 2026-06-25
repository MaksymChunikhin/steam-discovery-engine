import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_model(name: str = "all-MiniLM-L6-v2", device: str = None) -> SentenceTransformer:
    # Load a Sentence Transformer model (on GPU if a device is given)
    # Загружаем модель Sentence Transformer (на GPU, если задан device)
    return SentenceTransformer(name, device=device)


def encode_texts(texts, model: SentenceTransformer, batch_size: int = 256) -> np.ndarray:
    # Encode texts into L2-normalized embeddings
    # Кодируем тексты в L2-нормированные эмбеддинги
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)


def build_game_embeddings(reviews: pd.DataFrame, model: SentenceTransformer,
                          max_reviews_per_game: int = 200,
                          item_col: str = "game_id", text_col: str = "review",
                          playtime_col: str = "playtime_forever",
                          batch_size: int = 256, random_state: int = 42):
    # Sample up to N reviews per game, encode them, and pool into one
    # playtime-weighted (log1p of hours) embedding per game.
    # Сэмплируем до N отзывов на игру, кодируем и агрегируем в один эмбеддинг игры,
    # взвешивая по log1p времени игры (в часах).
    shuffled = reviews.sample(frac=1, random_state=random_state)
    sampled = shuffled.groupby(item_col, group_keys=False).head(max_reviews_per_game)

    emb = encode_texts(sampled[text_col].tolist(), model, batch_size=batch_size)
    weights = np.log1p(sampled[playtime_col].to_numpy(dtype=np.float64) / 60.0)  # hours
    game_ids = sampled[item_col].to_numpy()

    # Group row indices per game
    # Группируем индексы строк по игре
    idx_by_game = {}
    for i, g in enumerate(game_ids):
        idx_by_game.setdefault(g, []).append(i)

    out_ids, vecs, n_reviews = [], [], []
    for g, idxs in idx_by_game.items():
        w = weights[idxs]
        wsum = w.sum()
        if wsum <= 0:
            vec = emb[idxs].mean(axis=0)
        else:
            vec = (emb[idxs] * w[:, None]).sum(axis=0) / wsum
        out_ids.append(g)
        vecs.append(vec)
        n_reviews.append(len(idxs))

    matrix = np.vstack(vecs).astype(np.float32)
    matrix /= (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)  # L2 normalize
    meta = pd.DataFrame({
        "game_id": out_ids,
        "embedding_idx": range(len(out_ids)),
        "n_reviews": n_reviews,
    })
    return matrix, meta
