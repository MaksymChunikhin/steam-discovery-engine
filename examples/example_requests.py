"""Example requests against the running recommendation service.

Start the API first:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Then run:
    python examples/example_requests.py

Or use curl:
    curl http://localhost:8000/health
    curl http://localhost:8000/recommend/76561197960268765
    curl http://localhost:8000/similar/730
    curl http://localhost:8000/game/730

Примеры запросов к запущенному сервису рекомендаций (см. команды выше).
"""
import sys
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def show(title, resp):
    print(f"\n=== {title} ===")
    print(resp.status_code)
    print(resp.json())


def main():
    show("health", requests.get(f"{BASE}/health"))
    show("root", requests.get(f"{BASE}/"))

    # Pick a real user id from the service metadata
    # Берём реальный user_id из метаданных сервиса
    import numpy as np
    from pathlib import Path
    user_ids = np.load(Path(__file__).resolve().parents[1] / "models" / "metadata" / "user_ids.npy",
                       allow_pickle=True)
    uid = int(user_ids[0])

    show(f"recommend/{uid}", requests.get(f"{BASE}/recommend/{uid}"))
    show("similar/730", requests.get(f"{BASE}/similar/730"))
    show("game/730", requests.get(f"{BASE}/game/730"))


if __name__ == "__main__":
    main()
