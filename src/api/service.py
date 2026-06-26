from .loader import RecommendationPipeline

# Singleton pipeline: artifacts are loaded once at startup, never per request.
# Синглтон пайплайна: артефакты грузятся один раз при старте, не на каждый запрос.
_pipeline = None


def get_pipeline() -> RecommendationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RecommendationPipeline.load()
    return _pipeline
