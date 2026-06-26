from pydantic import BaseModel


class Recommendation(BaseModel):
    game_id: int
    title: str
    score: float
    reason: str
    genres: str


class RecommendationResponse(BaseModel):
    user_id: int
    n_recommendations: int
    recommendations: list[Recommendation]


class SimilarGame(BaseModel):
    game_id: int
    title: str
    score: float


class SimilarResponse(BaseModel):
    game_id: int
    game: str
    similar: list[SimilarGame]


class GameInfo(BaseModel):
    game_id: int
    title: str
    genres: str
    tags: str
    price: float


class DemoUser(BaseModel):
    display_name: str
    user_id: int
    favorite_genre: str
    games: int
    avg_playtime: float


class DemoGame(BaseModel):
    game_id: int
    title: str
    genres: str


class DemoResponse(BaseModel):
    users: list[DemoUser]
    games: list[DemoGame]
