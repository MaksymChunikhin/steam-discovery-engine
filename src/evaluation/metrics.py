import numpy as np

# Ranking metrics for top-k recommendation. Each function takes a ranked list of
# recommended item ids and a set of relevant (ground-truth) item ids.
# Ранжирующие метрики для top-k. Каждая функция принимает ранжированный список
# рекомендаций и множество релевантных (ground-truth) игр.


def recall_at_k(recommended, relevant, k: int) -> float:
    # Share of relevant items that appear in the top-k
    # Доля релевантных игр, попавших в top-k
    if not relevant:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended, relevant, k: int) -> float:
    # Average precision at k
    # Average precision на k
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for i, r in enumerate(recommended[:k]):
        if r in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k: int) -> float:
    # Normalized discounted cumulative gain at k (binary relevance)
    # Normalized DCG на k (бинарная релевантность)
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2)
              for i, r in enumerate(recommended[:k]) if r in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg
