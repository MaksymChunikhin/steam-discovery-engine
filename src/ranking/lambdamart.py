import lightgbm as lgb


def train_reranker(X, y, groups, n_estimators: int = 300, learning_rate: float = 0.05,
                   num_leaves: int = 31, random_state: int = 42):
    # LightGBM LambdaMART re-ranker; query groups = users, optimized for NDCG.
    # Re-ranker LightGBM LambdaMART; query-группы = пользователи, оптимизация под NDCG.
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        importance_type="gain",
        verbose=-1,
    )
    model.fit(X, y, group=groups, eval_at=[10])
    return model
