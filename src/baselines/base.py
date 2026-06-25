from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    # Common interface for every recommender so that evaluate() stays model-agnostic.
    # Общий интерфейс всех рекомендеров, чтобы evaluate() не зависел от конкретной модели.

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "BaseRecommender":
        # Train the model on the train interactions.
        # Обучаем модель на train-взаимодействиях.
        ...

    @abstractmethod
    def recommend(self, user_ids, k: int = 10) -> dict:
        # Return {user_id: [game_id, ...]} of length up to k, excluding train items.
        # Возвращаем {user_id: [game_id, ...]} длиной до k, исключая train-игры.
        ...
