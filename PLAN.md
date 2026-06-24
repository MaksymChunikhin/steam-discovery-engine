# Steam Game Recommendation System

**Two-Tower Retrieval & Learning-to-Rank**

Production-style recommendation system: NLP review embeddings → Two-Tower retrieval →
FAISS vector search → LambdaMART ranking → FastAPI serving → Docker deployment.

Оценивается через Recall@10, MAP@10, NDCG@10 против popularity, content-based и ALS baselines.

---

## Зафиксированные решения

- **Название проекта (заголовок/резюме):** `Steam Game Recommendation System` (тех-подзаголовок про Two-Tower & LTR — в README).
- **Имя репозитория:** `steam-discovery-engine`.
- **Ноутбуки:** 6 штук (по фазам) + `notebooks/README.md` как индекс. Переиспользуемый код — в `src/`, ноутбук импортирует из `src`.
- **`PLAN.md` коммитим в репо** — показывает, что проект шёл по плану, а не вырос из одного ноутбука.
- **Split:** leave-last-out per user (последнее взаимодействие каждого юзера в test). **Никогда random.**
- **Two-Tower negatives:** positive pairs + in-batch negatives (PyTorch). Не sampled softmax, не triplet.
- **Ranker:** LightGBM LambdaMART (`LGBMRanker`, `objective="lambdarank"`) — называть так в коде и README.
- **Thin deployable slice** (FastAPI + Docker вокруг baseline) — на неделе 2.5–3, НЕ уползает в конец.
- **Срок:** 5 недель (реалистичный график, который можно сжать).

---

## План по неделям

### Неделя 0 — Setup
- Скелет репозитория, `requirements.txt`, README-заготовка.
- Выбор датасета (Steam Games + Steam Reviews с Kaggle).
- Фиксация масштаба сэмплирования (top N игр по числу отзывов, max M отзывов на игру для эмбеддингов).
- **Ship:** репозиторий с пустыми модулями, данные в `data/raw`.

### Неделя 1 — Данные + EDA + interactions
- Очистка данных.
- EDA: жанры / теги / цены / распределение активности юзеров / длина и тональность отзывов / частые слова.
- Сборка таблицы `user_id, game_id, target` (`recommended` или `log(hours_played)`).
- **Ship:** `data/processed/interactions.parquet` + notebook с EDA.

### Неделя 1.5 — Data Validation (GATE) 🆕
Короткий, но обязательный этап. Проверяем ДО всего остального:
- `users >= 1000`, `games >= 1000`, `interactions >= 100k`
- sparsity
- cold-start % (юзеры/игры с < N взаимодействий)
- **Здесь же фиксируем split:** leave-last-out per user.
- Если пороги не проходят → меняем сэмплирование СЕЙЧАС, а не после обучения Two-Tower.
- **Ship:** короткий validation-отчёт (counts, sparsity, cold-start).

### Неделя 2 — Baselines + eval harness
- Popularity baseline.
- Content-based (genre/tag overlap + опционально TF-IDF/эмбеддинги, cosine similarity).
- ALS (`implicit`).
- Единый eval harness: Recall@10, NDCG@10, MAP@10 на leave-last-out split.
- **Ship:** таблица метрик с 2–3 baseline-моделями.

### Неделя 2.5–3 — Thin deployable slice
- FastAPI + Docker вокруг лучшего baseline (вероятно ALS).
- `GET /recommend/{user_id}` → список рекомендаций.
- **Ship:** рабочий контейнеризованный сервис на реальной модели. Можно показать на любом этапе дальше.

### Неделя 3–4 — NLP-эмбеддинги + Two-Tower + FAISS
- Sentence-transformer эмбеддинги отзывов (mean pooling, на сэмплированном поднаборе) → embedding игры (384/768).
- Two-Tower:
  - User tower: любимые жанры/теги, среднее время игры → Dense → user embedding.
  - Item tower: жанры, теги, review embedding → Dense → game embedding.
  - Обучение: positive pairs + in-batch negatives (PyTorch).
- FAISS индекс → top-100 кандидатов по `recommend(user_id)`.
- **Ship:** Two-Tower обгоняет baseline по Recall@10; FAISS подключён к тому же API-контракту.

### Неделя 4–5 — Ranking
- Candidate dataset: positives + sampled negatives, `user, candidate_game, label`.
- Фичи: `cosine(user_emb, game_emb)`, genre_overlap, tag_overlap, game_popularity, review_score, price.
- Модель: LightGBM LambdaMART (`LGBMRanker`, `objective="lambdarank"`) → top-10.
- **Ship:** ranker улучшает NDCG@10 относительно retrieval-only.

### Неделя 5 — Полировка + отчёт
- Rule-based explainability: "Recommended because: похоже на X / общие теги / позитивные отзывы".
- **`reports/final_report.md`** 🆕: датасет → методология → метрики → сравнение моделей → выводы.
- README: архитектурная схема + пример запроса/ответа API.
- **Ship:** финальный README + сводная таблица метрик.

### Стретч (если останется время)
- SHAP-объяснения для ranker.
- Streamlit-демо UI.
- Diversity-constraints в ре-ранкинге.

---

## Итоговая таблица сравнения (цель eval)

| Модель               | Recall@10 | NDCG@10 | MAP@10 |
|----------------------|-----------|---------|--------|
| Popularity           |           |         |        |
| Content-Based        |           |         |        |
| ALS                  |           |         |        |
| Two-Tower            |           |         |        |
| Two-Tower + Ranker   |           |         |        |

---

## Структура репозитория

```
steam-discovery-engine/
├── PLAN.md
├── README.md
├── notebooks/
│   ├── README.md                    # индекс: номер → ноутбук → что внутри → что на выходе
│   ├── 01_eda.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_embeddings.ipynb
│   ├── 05_two_tower.ipynb
│   └── 06_ranking.ipynb
├── src/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── retrieval/
│   ├── ranking/
│   ├── api/
│   └── utils/
├── configs/
├── models/
├── reports/
│   └── final_report.md
├── data/
│   ├── raw/
│   └── processed/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## API контракт

```
GET /recommend/123

{
  "user_id": 123,
  "recommendations": [
    { "game": "Cyberpunk 2077", "score": 0.94 }
  ]
}
```
