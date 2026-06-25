import pandas as pd


def leave_last_out(interactions: pd.DataFrame,
                   user_col: str = "user_id",
                   time_col: str = "unix_timestamp_created",
                   item_col: str = "game_id"):
    # Hold out each user's most recent interaction as test, keep the rest as train.
    # Ties on time are broken deterministically by item_col, so the split is reproducible.
    # Для каждого пользователя последнее по времени взаимодействие -> test, остальное -> train.
    # Совпадения по времени разрешаются детерминированно по item_col -> split воспроизводим.
    order = interactions.sort_values([user_col, time_col, item_col])
    is_last = order.groupby(user_col, sort=False).cumcount(ascending=False) == 0
    train = order[~is_last]
    test = order[is_last]
    return train, test
