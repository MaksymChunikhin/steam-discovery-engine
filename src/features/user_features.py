import pandas as pd


def hours_played(playtime_minutes: pd.Series) -> pd.Series:
    # Convert Steam playtime from minutes to hours
    # Переводим время игры Steam из минут в часы
    return playtime_minutes / 60.0
