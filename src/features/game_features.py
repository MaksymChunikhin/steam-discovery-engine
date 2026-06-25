import pandas as pd


def _split_csv(series: pd.Series) -> pd.Series:
    # Split a comma-separated string column into a list of clean tokens
    # Разбиваем колонку строк через запятую в список очищенных значений
    return (series.fillna("")
                  .str.split(",")
                  .apply(lambda xs: [x.strip() for x in xs if x.strip()]))


def split_genres(genres: pd.Series) -> pd.Series:
    # Split the Genres column into lists
    # Разбиваем колонку Genres в списки
    return _split_csv(genres)


def split_tags(tags: pd.Series) -> pd.Series:
    # Split the Tags column into lists
    # Разбиваем колонку Tags в списки
    return _split_csv(tags)


def count_tags(tags: pd.Series) -> pd.Series:
    # Count the number of tags per game
    # Считаем число тегов на игру
    return split_tags(tags).apply(len)
