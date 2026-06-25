def fuse_candidates(retrievers: dict, user_ids, k_per: int = 30) -> dict:
    # Multi-source candidate fusion: union of each retriever's top-k_per per user.
    # Объединение кандидатов: union top-k_per от каждого ретривера на юзера.
    user_ids = list(user_ids)
    per_source = {name: rec.recommend(user_ids, k=k_per) for name, rec in retrievers.items()}

    fused = {}
    for u in user_ids:
        seen, ordered = set(), []
        for name in retrievers:
            for g in per_source[name].get(u, []):
                if g not in seen:
                    seen.add(g)
                    ordered.append(g)
        fused[u] = ordered
    return fused
