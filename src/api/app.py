from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .service import get_pipeline
from .schemas import (RecommendationResponse, SimilarResponse, GameInfo,
                      DemoResponse)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all artifacts once at startup (offline-trained, inference only)
    # Грузим все артефакты один раз при старте (обучено оффлайн, только инференс)
    get_pipeline()
    yield


app = FastAPI(title="Steam Discovery Engine",
              description="Two-stage game recommendation service (retrieval + re-ranking).",
              version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    p = get_pipeline()
    return {
        "service": "Steam Discovery Engine",
        "version": "1.0.0",
        "pipeline": "Two-Stage Recommendation Pipeline",
        "models_loaded": True,
        "users": len(p.ctx["user_ids"]),
        "games": len(p.ctx["item_ids"]),
        "docs": "/docs",
        "demo": "/demo",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, k: int = 10):
    p = get_pipeline()
    if not p.has_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    recs = p.recommend(user_id, k=k)
    return {"user_id": user_id, "n_recommendations": len(recs), "recommendations": recs}


@app.get("/similar/{game_id}", response_model=SimilarResponse)
def similar(game_id: int, k: int = 10):
    p = get_pipeline()
    result = p.similar(game_id, k=k)
    if result is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return {"game_id": game_id, "game": p._title(game_id), "similar": result}


@app.get("/game/{game_id}", response_model=GameInfo)
def game(game_id: int):
    p = get_pipeline()
    info = p.game_info(game_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return info


@app.get("/demo", response_model=DemoResponse)
def demo():
    # Demo users and featured games so clients need no hard-coded IDs
    # Демо-юзеры и витрина игр, чтобы клиентам не нужны были захардкоженные ID
    p = get_pipeline()
    return {"users": p.demo["users"], "games": p.demo["games"]}


@app.get("/demo/users")
def demo_users():
    return get_pipeline().demo["users"]


@app.get("/demo/games")
def demo_games():
    return get_pipeline().demo["games"]
