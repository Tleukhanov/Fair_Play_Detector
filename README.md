# Fair Play Detector

Сервис вероятностного анализа публичных партий Lichess на признаки
машиноподобной игры.

**Важно о позиционировании:** результат — это статистический сигнал для
дополнительной проверки, а **не** доказательство читерства и не основание для
автомодерации. Так зафиксировано в API-контрактах, в дисклеймерах и в
`packages/llm/fallback.py`.

---

## 1. Что реально работает сегодня

| Слой | Состояние | Файлы |
|---|---|---|
| **Схема БД** | ✅ 6 таблиц, 5 миграций Alembic | `apps/api/app/db/models.py`, `migrations/` |
| **API сканирования** | ✅ 5 эндпоинтов, валидация, 202/404/409/503 | `apps/api/app/api/routes.py` |
| **Оркестрация** | ✅ state machine + lease + stale-scan reaper | `apps/api/app/workers/state_machine.py`, `store.py`, `service.py` |
| **Фоновый исполнитель** | ✅ 2 параллельно, 20 в очереди | `apps/api/app/workers/executor.py` |
| **Ingest Lichess** | ✅ стриминг PGN, retry, часы (`clocks=true`), per-scan snapshot | `packages/ingest/` |
| **Stockfish-адаптер** | ✅ подключён, CPL в POV ходящего, клип мата, часы, depth-1/9/15 | `packages/chess_analysis/` |
| **Анализ + скоринг** | ✅ `ChessAnalysisStage` + `ScoringStage` подключены | `apps/api/app/workers/analysis_stages.py` |
| **ML-модель** | ✅ обучена на ChessFraud, артефакт + `TrainedScorer` | `packages/ml/train.py`, `scorer.py` |
| **LLM-слой** | ⚠️ только детерминированный fallback | `packages/llm/` |
| **Frontend** | ❌ отсутствует | — |
| **Maia** | ❌ не подключена (не входит в inference) | — |

Проверки в локальном окружении: `pytest` (см. раздел 8), `ruff` чистый,
`mypy` — см. актуальный вывод в CI/локально.

---

## 2. Данные: только ChessFraud

`Games.csv` (Kaggle «Spotting Cheaters») **не используется**: это синтетические
игры ботов Maia vs Stockfish, без реальных людей и без часов. Он не даёт
переносимости на реальных игроков.

Используются две конфигурации
[`artemlepin/chess-fraud`](https://huggingface.co/datasets/artemlepin/chess-fraud)
(CC BY 4.0):

| Конфиг | Роль | Строк | `is_used` | Игроков | Игр |
|---|---|---:|---:|---:|---:|
| `chess_fraud/full` | реальный контролируемый турнир, **обучение и оценка** | 38 510 | 28 410 | 49 | 1 010 player-games |
| `chess_fraud_synth/train` | синтетика для будущих экспериментов | 860 183 | 334 140 | 960 | 9 600 |
| `chess_fraud_synth/test` | player-disjoint тест синтетики | 214 104 | 83 067 | 240 | 2 400 |

Факты по `chess_fraud`:

- 505 партий, 1 010 player-side игр, 49 игроков, контроль `300+0`;
- 40.3% player-games с подсказками, 24.4% ходов с подсказками;
- evals/CPL/normalized CPL/часы/Maia-2/Allie уже посчитаны (Stockfish 17, depth 15);
- метка `is_cheating_move` — ход следует за показанной линией Stockfish;
  `is_cheating_player_game` — в игре была такая серия.

Данные **не коммитятся**. Загрузка:

```bash
python scripts/download_chessfraud.py \
  --output-dir /home/yeraly/Data/fair_play/chessfraud
```

Скрипт пинует revision HuggingFace, считает sha256 и пишет `manifest.json`.
Читать данные можно через `FAIR_PLAY_DATA_DIR`:

```bash
export FAIR_PLAY_DATA_DIR=/home/yeraly/Data/fair_play/chessfraud
```

---

## 3. Фичи и антиутечка

Контракт фич — `packages/ml/feature_contract.py` (12 колонок, версия `2.0.0`):

```
centipawn_loss, normalized_centipawn_loss,
match_sf1, match_sf9, match_sf15,
eval_swing, abs_eval, phase,
move_thinking_time, time_ratio,
player_elo_centered, opponent_elo_centered
```

Правила, которые проверяются кодом (fail-closed):

- **Split только по игроку**, никогда по строкам.
- **Чёрный список** меток/идентичностей/сырых значений: `is_cheating_*`,
  `player_hint_shown`, `assistance_*`, `move_player`, `position_*`, `game_id`,
  `player_id`, `move_stockfish_*`, `player_elo`, `eval_*`, `half_move` и т.д.
  Если такая колонка попала в матрицу — обучение падает.
- **Elo только как нормализатор** (`player_elo_centered`), не как сырая фича,
  чтобы модель не выучила «кто в когорте читерит».
- **Дебют отсекается** (`half_move >= 21`), как в ChessFraud.
- **Никакой сырой доски/ходов** в модели: FEN/UCI используются только для
  расчёта `match_*` и CPL.
- **Parity training/serve**: один модуль `packages/ml/features.py` и один
  `aggregate_game_features`; на serve-пути те же формулы, что в parquet.

---

## 4. Модель и честные метрики

**Serving-модель — game-level** (не per-move): она принимает агрегаты одной
player-side игры и предсказывает вероятность «в этой игре была подсказка».
Player-level риск — среднее по играм скана + bootstrap-CI.

Почему не per-move в проде: текущий `ScoringStage`/`ScoringInput` работает с
агрегатами `GameAggregateFeatures`; per-move модель требует расширения схемы и
протаскивания вероятностей по каждому ходу. Per-move модель обучена и лежит в
отчёте как research-артефакт, но в serving не подключена.

**Данные:** `chess_fraud/full`, 1 010 player-games, 49 игроков, prevalence 40.3%.
**Split:** player-disjoint 60/20/20 (seed 2025), 610 / 202 / 198 игр,
29 / 10 / 10 игроков.

**Модель:** XGBoost (300 деревьев, depth 4), сигмоидная калибровка на
validation, артефакт `data/models/chessfraud-v1`.

**Test (198 игр, 82 positive, prevalence 41.4%):**

| Метрика | Значение | Player-bootstrap 95% CI |
|---|---:|---|
| ROC-AUC | **0.7303** | 0.6454–0.8130 |
| PR-AUC | **0.7187** | 0.6327–0.8270 |
| Brier | **0.2005** | 0.1755–0.2294 |
| Recall @ FPR 1% | **0.1098** | 0.0588–0.1769 |
| Recall @ FPR 5% | **0.1707** | 0.1000–0.2746 |

Пороги выбраны только на validation:
`suspicious = 0.7428` (FPR 5%), `high_risk = 0.7764` (FPR 1%).

**Baselines (test):**

| Score | ROC-AUC | PR-AUC | Recall @5% FPR |
|---|---:|---:|---:|
| `top_1_match_rate` | 0.7265 | 0.6874 | 0.1098 |
| inverse mean CPL | 0.5299 | 0.4235 | 0.0122 |
| `BaselineScorer` (player-level, 10 игроков) | 0.5833 | — | — |
| Trained model (player-level, те же 10) | 0.7500 | — | — |

**Честный вывод:** преимущество над простой эвристикой `top_1_match_rate`
**не доказано**: paired player-bootstrap `AUC(model) − AUC(top1) = −0.00013`,
CI `[−0.0598, +0.0556]`. То есть модель пока не лучше «доли совпадений с top-1»
на этой когорте. Числа `precision` при prevalence 40% **не переносятся** на
реальный трафик (<1%); сопоставимая величина — recall при фиксированном FPR.

**Per-move research** (GroupKFold 3 by player, 28 410 строк): OOF ROC-AUC
**0.6639**, PR-AUC **0.4262**. В serving не подключена.

---

## 5. Как устроен скан

```
POST /api/v1/scans
  -> queued
  -> fetching   Lichess PGN (clocks=true) -> games (per-scan snapshot)
  -> analyzing  Stockfish: mover-POV CPL, top1 depth 1/9/15, часы, forced/noisy
  -> scoring    TrainedScorer (или BaselineScorer, если артефакта нет)
  -> explaining deterministic fallback
  -> completed  verdict + per-game scores + moves
```

Особенности:

- партии хранятся **per-scan** (`UNIQUE(lichess_game_id, scan_id)`), повторный
  скан того же игрока не теряет игры;
- кэш фич по `cache_key` переиспользует анализ и копирует его в строку текущей
  игры;
- `game_scores` уникальны по `(game_id, model_version)`, re-score идемпотентен;
- зависшие сканы старше `STALE_SCAN_MINUTES` переводятся в `failed` на старте;
- `rated_only` доезжает до Lichess API.

---

## 6. Что НЕ работает / ограничения

- **Per-move модель не в serving.** В проде game-level; per-move — research.
- **Maia/Allie не подключены** к inference: `maia_divergence` всегда `None`,
  поля DTO это допускают.
- **Маленькая когорта:** 49 игроков, test 10 игроков / 198 игр; широкие CI,
  split чувствителен к seed.
- **Нет доказанного прироста** над эвристикой top-1 на этой выборке.
- **Prevalence 40%** в ChessFraud; precision нельзя переносить на реальный
  трафик.
- **Одна ошибка движка валит весь scan** (`ANALYSIS_FAILED`); per-game isolation
  не сделана.
- **LLM-провайдер не подключён** — только deterministic fallback.
- **Frontend отсутствует.**
- **Нет auth/rate-limit/retention** на публичном API; перед публикацией нужен
  security/privacy review.
- `ScanPipeline` (`workers/pipeline.py`) не используется — прод-путь идёт через
  `ScanWorker`.
- Репозитории-протоколы (`domain/repositories.py`) без реализаций.

---

## 7. Структура репозитория

```
apps/api/app/          FastAPI: роуты, модели БД, воркеры
packages/
  ingest/              клиент Lichess, PGN, персистентность
  chess_analysis/      Stockfish + агрегация признаков
  feature_engineering/ контракт GameAggregateFeatures
  ml/                  feature contract, data loader, features, train, scorer
  dataset/             старый bounded collector (Lichess Elite/TCEC; не используется)
  llm/                 контракты объяснений + deterministic fallback
scripts/               download_chessfraud.py
migrations/            Alembic 0001…0005
tests/                 pytest
data/models/           артефакты (gitignored)
data_raw/              сырые parquet (gitignored)
```

**6 таблиц:** `users`, `scans`, `games`, `game_features`, `game_scores`,
`verdicts`. State machine:
`queued → fetching → analyzing → scoring → explaining → completed`,
в `failed` можно уйти из любого нетерминального состояния.

---

## 8. Как запустить

```bash
python -m pip install -e ".[dev,dataset,ml]"
cp .env.example .env
alembic upgrade head
uvicorn apps.api.app.main:app --reload
```

Нужен Stockfish. Путь задаётся `STOCKFISH_PATH` (по умолчанию
`/usr/games/stockfish`, иначе `stockfish` из `PATH`):

```bash
which stockfish        # /home/yeraly/.local/bin/stockfish
export STOCKFISH_PATH=$(which stockfish)
```

Проверка: `GET /health` → `{"status": "ok"}`, Swagger на `/docs`.

### Данные и обучение

```bash
python scripts/download_chessfraud.py \
  --output-dir /home/yeraly/Data/fair_play/chessfraud

export FAIR_PLAY_DATA_DIR=/home/yeraly/Data/fair_play/chessfraud
python -m packages.ml.train \
  --data-dir "$FAIR_PLAY_DATA_DIR" \
  --output-dir data/models/chessfraud-v1
```

`MODEL_DIR` (по умолчанию `data/models/chessfraud-v1`) подхватывается API:
если `metadata.json` есть — используется `TrainedScorer`, иначе
`BaselineScorer`.

### Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/api/v1/scans` | Запустить сканирование → `202` + `scan_id`, либо `503 QUEUE_FULL` |
| `GET` | `/api/v1/scans/{id}` | Прогресс: статус, найдено/проанализировано, ошибка |
| `GET` | `/api/v1/scans/{id}/verdict` | Вердикт (или `409`, пока не `completed`) |
| `GET` | `/api/v1/scans/{id}/games` | Партии постранично, с реальными агрегатами |
| `GET` | `/api/v1/scans/{id}/games/{game_id}` | Детали игры + moves |

Ограничения запроса: ник 1–30 символов, `max_games` 10–50, только
`blitz|rapid`, `rated_only` учитывается.

### Тесты и линтеры

```bash
python -m pytest -q                          # см. актуальный вывод
python -m ruff check apps packages tests scripts
python -m mypy
```

Stockfish-тесты пропускаются, если бинарь не найден.

> ⚠️ В репозитории нет `.venv`; используй локальное окружение Python 3.12+
> с `.[dev,dataset,ml]`. Системный Python без зависимостей тесты не запустит.

---

## 9. Бэклог по приоритету

1. **Per-move serving**: протащить вероятность по каждому ходу в
   `GameAggregateFeatures` и `ScoringStage`; оценить, даёт ли это прирост.
2. **Maia-2 на Elo игрока** как human-aligned сигнал (сейчас `maia_divergence`
   пуст).
3. **Больше реальных данных**: ChessFraud-Synth как pretrain → fine-tune на
   `chess_fraud`, внешняя валидация на новых когортах.
4. **Per-game isolation**: одна плохая партия не должна валить scan.
5. **LLM-провайдер** (Gemini Flash) вместо fallback, со strict schema.
6. **Frontend** (Next.js) + polling + графики CPL/time/top-1.
7. **Security/privacy**: auth, rate-limit, retention, запрет индексации.
8. **PostgreSQL/Redis** для multi-process worker.

---

## 10. Лицензии и атрибуция

ChessFraud распространяется по CC BY 4.0. Если используешь данные/модель,
цитируй:

```
Linich, A., Lepin, A., Sakhovskiy, A., Toleutaeva, A., Lepa, G.,
Neznamov, A., Budennyy, S. (2026).
ChessFraud: Exploring the Capabilities of Human-Aligned Models for Cheating
Detection in Online Chess. KDD 2026. https://doi.org/10.1145/3770855.3817587
```

ChessFraud-Synth — производное от публичной Lichess DB (CC0).
