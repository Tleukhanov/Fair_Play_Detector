# Fair Play Detector

Сервис вероятностного анализа публичных партий Lichess на признаки машиноподобной игры.

**Важно о позиционировании:** результат — это статистический сигнал для дополнительной проверки,
а **не** доказательство читерства и не основание для автомодерации. Так зафиксировано в коде
(`packages/llm/fallback.py`), в API-контрактах и в дисклеймерах.

---

## 1. Что реально работает сегодня

| Слой | Состояние | Файлы |
|---|---|---|
| **Схема БД** | ✅ 6 таблиц, 4 миграции Alembic | `apps/api/app/db/models.py`, `migrations/` |
| **API сканирования** | ✅ 5 эндпоинтов, валидация, 202/409/503 | `apps/api/app/api/routes.py` |
| **Оркестрация** | ✅ state machine + lease-защита от двойного захвата | `apps/api/app/workers/state_machine.py`, `store.py`, `service.py` |
| **Фоновый исполнитель** | ✅ лимиты: 2 параллельно, 20 в очереди | `apps/api/app/workers/executor.py` |
| **Ingest Lichess** | ✅ стриминг PGN, retry с `Retry-After`, дедупликация | `packages/ingest/` |
| **Адаптер Stockfish** | ✅ написан, **не подключён** | `packages/chess_analysis/stockfish.py` |
| **Сбор датасета** | ✅ гейт 5 GiB, Parquet, отчёт | `packages/dataset/` |
| **Обучение ML** | ✅ 2 модели, калибровка, антипротечка | `packages/ml/` |
| **LLM-слой** | ⚠️ только детерминированный fallback | `packages/llm/` |

Проверено: **17/17 тестов проходят**, `ruff` чистый.

---

## 2. Что НЕ работает (заглушки) — читать обязательно

### 2.1 Анализ и скоринг не подключены 🔴

`apps/api/app/workers/default_stages.py` отдаёт **реальную** стадию только для скачивания:

```python
local_scan_stages(...) → (LichessFetchStage, LocalNoOpStage, LocalNoOpStage, LocalNoOpStage)
```

При этом `ChessAnalysisStage` и `ScoringStage` в `analysis_stages.py` **написаны, но не вызваны нигде**.

**Последствие:** сканирование скачает партии и на этом остановится. Таблицы `game_features`,
`game_scores`, `verdicts` останутся пустыми, а `GET /verdict` будет отдавать **409**, а не 200.

**Как чинить:** в `default_stages.py` заменить три `LocalNoOpStage` на `ChessAnalysisStage`
и `ScoringStage`, создав `StockfishEngine` в `_create_scan_job()` (`apps/api/app/main.py:20-28`).

### 2.2 API возвращает нули вместо данных 🔴

`_game_to_dto()` (`routes.py:190-202`) жёстко возврачает `move_count=0`,
`top_move_match_rate=0.0`, `mean_centipawn_loss=0.0`, `maia_divergence=0.0`,
а `GET .../games/{game_id}` отдаёт `moves=[]` — **хотя колонки в БД есть**.
Нужно подтягивать `GameScoreModel` и `GameFeaturesModel`.

### 2.3 Время на ход не извлекается 🟡

`StockfishEngine` не парсит clocks из PGN, поэтому `MoveAnalysis.time_seconds` всегда `None`,
а значит `identical_time_share`, `median_time_seconds`, `time_variance` — пустые.
Это **20% веса** baseline-формулы (`REGULAR_MOVE_TIMES`) и один из 4 признаков для ML.

### 2.4 Maia не подключена 🟡

Поля `maia_eval`, `maia_move_probability`, `maia_divergence` всегда `None`,
поэтому `maia_divergence` в признаках пуст, а его 15% веса в baseline дают ноль.

### 2.5 Сборщик датасета не считает поведенческие признаки 🔴

`packages/dataset/collector.py::_to_feature_row` оставляет `mean_cpl`, `top1_match_rate`,
`top3_match_rate`, `mean_move_time_ms` пустыми — он не гоняет Stockfish по собранным партиям.

**Последствие:** `packages/ml/cli.py` на этих данных **падает** с
`ValueError: behavioral features are incomplete; run Stockfish analysis before training`.
Это **задуманное поведение** (fail-closed), а не поломка: обучение на неполных признаках
должно быть невозможно.

### 2.6 Прочее

- `ScanPipeline` (`workers/pipeline.py`) не используется — прод-путь идёт через `ScanWorker`
- Репозитории-протоколы (`domain/repositories.py`) объявлены, реализаций нет
- `rated_only` из запроса игнорируется (всегда `True`)
- `current_stage` в статусе — копия `status`, отдельного поля нет
- `difficult_position_match_rate` = `top_1_rate` (заглушка)
- `REDIS_URL` объявлен, но не используется; воркеры не переживают рестарт процесса
- Frontend отсутствует

---

## 3. Почему метрики ML 0.999 — это брак, а не победа

В `FAIR_PLAY_DETECTOR_ML_REPORT.md` зафиксирован ROC-AUC **0.99933** на 350k строк.
Это не точность модели, а утечка: первый датасет (Lichess Elite vs TCEC) различался
по метаданным источника, а не по стилю игры. Пришлось выкинуть фичи `rated`, `speed`,
`result`, `move_count`.

Агент починил это в коммите `a899408 fix(ml): reject source leakage in training gate` —
теперь обучение **fail-closed** отказывается на:
- строках вне `move_count` 30..80 или чужом `speed`
- данных без PGN-колонки
- классах с покрытием поведенческих признаков < 95%

**Правило, которое стоит запомнить:** результат 0.99+ на первой попытке = ищи лик,
а не празднуй.

---

## 4. Новый датасет `Games.csv` (48k партий) — ещё не подключён

Локально лежит `Games.csv` (27 МБ, в git не добавлен):

| Колонка | Смысл |
|---|---|
| `Game` | партия в нотации |
| `Liste cheat white` / `black` | битовая маска помеченных ходов |
| `Elo White` / `Elo Black` | рейтинги |
| `Score` | результат |

~40% партий с метками, ~19% полностью чистых. Это **настоящая разметка per-move**
вместо прокси-классификации «источник данных».

**Что нужно сделать:** парсер нотации → FEN, маски → per-move метки, признаки через Stockfish.
Задача формулируется как «этот ход сыгран человеком или движком», метрики —
precision/recall по помеченным ходам. **Время на ход недоступно — исключить из MVP.**

---

## 5. Структура репозитория

```
apps/api/app/          FastAPI: роуты, модели БД, воркеры
packages/
  ingest/              клиент Lichess, парсинг PGN, персистентность
  chess_analysis/      адаптер Stockfish + агрегация признаков
  feature_engineering/ версионированный контракт признаков
  dataset/             сбор датасета (Parquet) + гейт + отчёт
  ml/                  обучение, baseline-скорер, калибровка
  llm/                 контракты объяснений + детерминированный fallback
migrations/            Alembic: 0001…0004
tests/                 6 файлов, 17 тестов
```

**6 таблиц БД:** `users`, `scans`, `games`, `game_features`, `game_scores`, `verdicts`.
Уникальность `games` — по `(lichess_game_id, user_id)`, у `game_features` — по `cache_key`,
у `verdicts` — по `scan_id`.

**State machine скана:**
`queued → fetching → analyzing → scoring → explaining → completed`, в `failed` можно уйти
из любого нетерминального состояния. Захват скана атомарный (`lease_id`), двойной захват невозможен.

---

## 6. Как запустить

```bash
python -m pip install -e ".[dev]"
cp .env.example .env            # Windows: Copy-Item .env.example .env
alembic upgrade head
uvicorn apps.api.app.main:app --reload
```

Проверка: `GET /health` → `{"status": "ok"}`, Swagger на `/docs`.

### Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/api/v1/scans` | Запустить сканирование → `202` + `scan_id`, либо `503 QUEUE_FULL` |
| `GET` | `/api/v1/scans/{id}` | Прогресс: статус, найдено/проанализировано, ошибка |
| `GET` | `/api/v1/scans/{id}/verdict` | Вердикт (или `409`, пока не `completed`) |
| `GET` | `/api/v1/scans/{id}/games` | Партии постранично |
| `GET` | `/api/v1/scans/{id}/games/{game_id}` | Детали игры (⚠️ сейчас заглушка) |

Ограничения запроса: ник 1–30 символов, `max_games` 10–50, только `blitz|rapid`, только рейтинговые.

### Сбор датасета

```bash
python -m pip install -e ".[dataset]"
python -m packages.dataset.cli \
  --lichess-elite-month 2025-01 \
  --tcec-season 2024 \
  --output-dir data/processed
```

Оба периода **обязательны** — без них команда падает до сетевого запроса.
Лимит **5 GiB** на оба источника, при превышении — `DATASET_QUOTA_EXCEEDED`.
Команда **никогда не запускает обучение**: останавливается и ждёт подтверждения отчёта.

### Обучение

```bash
python -m pip install -e ".[ml]"
python -m packages.ml.cli \
  --input-dir data/processed \
  --report data/models/ml_training_report.json \
  --speed rapid
```

Модели: Logistic Regression (интерпретируемая) + XGBoost, обе с сигмоидной калибровкой.
Пороги подбираются на validation по максимуму `precision × recall`.
Метрики: ROC-AUC, PR-AUC, recall при FPR ≤ 1%, precision на `high_risk`, Brier score.

### Тесты и линтеры

```bash
python -m pytest -q                          # 17 passed
python -m ruff check apps packages tests     # All checks passed!
python -m mypy                               # strict = true
```

> ⚠️ Закоммиченный `.venv` **нерабочий** — нет `httpx`, `python-chess`, `pyarrow`, `xgboost`.
> Тесты проходят только в системном Python 3.14. Пересоздай окружение при необходимости.

---

## 7. Бэклог по приоритету

1. **Подключить `ChessAnalysisStage` и `ScoringStage`** — без этого сканирование не даёт вердикта
2. **Подключить `StockfishEngine`** в `_create_scan_job()`, починить `_game_to_dto`
3. **Извлечь время на ход** из PGN (clocks) — вернёт 20% веса baseline
4. **Переориентировать датасет на `Games.csv`** — настоящая per-move разметка
5. **Подключить LLM-провайдера** (Gemini Flash) вместо детерминированного fallback
6. **Maia** — восстановить `maia_divergence` (или убрать из модели)
7. Alembic-реализации репозиториев, вынести `REDIS_URL` в Phase 2

---

## 8. Конфигурация

| Переменная | По умолчанию | Ограничение |
|---|---|---|
| `APP_ENV` / `APP_NAME` / `LOG_LEVEL` | `development` / `fair-play-detector` / `INFO` | |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | порт 1..65535 |
| `LICHESS_API_BASE_URL` | `https://lichess.org` | HTTP-URL |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/fair_play.db` | |
| `ANALYSIS_DEPTH` | `15` | ≥ 1 |
| `ANALYSIS_TIME_MS` | `100` | ≥ 1 |
| `MAX_CONCURRENT_SCANS` | `2` | ≥ 1 |
| `MAX_PENDING_SCANS` | `20` | ≥ 0 |

`REDIS_URL` объявлен, но не используется. PostgreSQL и Redis — закомментированы в `.env.example` (Phase 2).