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
| **LLM-слой** | ✅ RAG-объяснение + обязательный fallback; live-провайдер ждёт ключ | `packages/llm/` |
| **Фронтенд** | ✅ статический no-build UI на `/app/` | `apps/web/` |
| **Maia** | ❌ не подключена (не входит в inference) | — |

Проверки в локальном окружении: **229 passed** (228 + smoke с локальным
артефактом модели и ChessFraud parquet), `ruff` и `mypy` чистые. На чистом
checkout без gitignored-артефактов — **228 passed, 1 skipped**; в CI столько же
(smoke-тест скипается без артефакта). CI-workflow: Python 3.12 + Stockfish +
ruff/mypy/pytest.

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

**Пороги вердикта (player-level).** `TrainedScorer` агрегирует per-game
вероятности в среднее по играм, поэтому вердикт использует пороги на шкале
player-mean, выбранные на validation по 10 игрокам:
`suspicious = 0.4915` (`mean + 1σ`), `high_risk = 0.5638` (`mean + 2σ`).
Per-game пороги `0.7428 / 0.7764` (FPR 5%/1%) сохранены в metadata только для
отчётности: смешивать их с player-mean нельзя — именно это давало вечный
`low_risk`.

На validation/test это даёт **8 low / 2 suspicious / 0 high**. `high_risk` в
текущей когорте почти недостижим (max mean 0.5630 при пороге 0.5638); пороги
построены по 10 игрокам, поэтому шумные — пересчитать на большей когорте.

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

Машиночитаемый отчёт с метриками, порогами и ограничениями:
`reports/chessfraud_metrics.json`.

---

## 5. LLM-объяснение

LLM **не выставляет вердикт**. Числовой слой (`TrainedScorer` или
`BaselineScorer`) считает риск-скор, интервал и вклады признаков; LLM только
объясняет по-русски, почему модель пришла к такому решению. Score, порог и
интервал из ответа LLM не меняются: на вход идёт уже посчитанный вердикт, на
выходе — только текст.

**Что ограничивает ответ (RAG + валидатор).** В промпт попадают вердикт и
посчитанные `evidence`/`contribution`/`direction`, плюс курируемая база знаний
`packages/llm/knowledge.json` (политики, пороги, ограничения). Валидатор
`packages/llm/validator.py` отклоняет ответ, если в нём:

- есть обвинение или запрещённая лексика (`читер`, `винов`, `доказано`, `бан`
  и т. п.);
- нет фразы «не является доказательством»;
- придуманы числа, которых нет во входных данных или источниках
  (например, «97% точность»);
- `reasoning` ссылается на неизвестный `evidence_code` или вклад не совпадает
  с вердиктом;
- нет обязательных источников `policy.no_accusation` / `policy.factual_only`.

После одной попытки исправления включается детерминированное объяснение; скан
при этом не падает.

**Приватность.** В провайдер уходит только `ExplanationInput`: уровень риска,
score, интервал, число партий, признаки и вклады, ограничения. Ник, PGN, ID
партий, имена соперников и рейтинги не передаются — в контракте
`packages/llm/contracts.py` для них нет полей.

**Без ключа.** Если `LLM_API_KEY` пуст, `LLM_ENABLED=false` или провайдер
недоступен, работает `DeterministicExplainer`: он строит объяснение из тех же
признаков, вкладов, интервала и размера выборки. В UI карточка «Почему такой
вердикт» показывает бейдж «без LLM» и, если backend вернул `fallback_reason`,
человекочитаемую причину (например, «Таймаут провайдера»); иначе — общую строку
«LLM недоступен — объяснение сформировано детерминированно». Ключи и сырые
ошибки провайдера в интерфейс не попадают.

**Переменные окружения** (ключ хранится в `.env`, который gitignored, и никогда
не коммитится):

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LLM_ENABLED` | `true` | Выключить слой, оставив fallback |
| `LLM_PROVIDER` | `openai` | `openai`, `openai_compatible` или `gemini` |
| `LLM_API_KEY` | пусто | Ключ провайдера; пусто → deterministic fallback |
| `LLM_MODEL` | `gpt-4o-mini` | Имя модели **выбранного** провайдера |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Endpoint; для Gemini — `.../v1beta` |
| `LLM_TIMEOUT_SECONDS` | `30` | Таймаут одного запроса |
| `LLM_MAX_RETRIES` | `1` | Повторы на 429/5xx |
| `LLM_TEMPERATURE` | `0.2` | Температура генерации |
| `LLM_MAX_OUTPUT_TOKENS` | `900` | Лимит ответа |
| `LLM_KNOWLEDGE_PATH` | пусто | Альтернативный `knowledge.json` |

Примеры провайдеров (модель должна соответствовать провайдеру: имя модели
OpenAI не сработает на Gemini endpoint):

```bash
# OpenAI-совместимый endpoint (по умолчанию)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1

# OpenRouter
LLM_PROVIDER=openai_compatible
LLM_MODEL=openai/gpt-4o-mini
LLM_BASE_URL=https://openrouter.ai/api/v1

# Google Gemini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

---

## 6. Как устроен скан

```
POST /api/v1/scans
  -> queued
  -> fetching   Lichess PGN (clocks=true) -> games (per-scan snapshot)
  -> analyzing  Stockfish: mover-POV CPL, top1 depth 1/9/15, часы, forced/noisy
  -> scoring    TrainedScorer (или BaselineScorer, если артефакта нет)
  -> explaining LLM-объяснение или deterministic fallback
  -> completed  verdict + per-game scores + moves
```

Особенности:

- партии хранятся **per-scan** (`UNIQUE(lichess_game_id, scan_id)`), повторный
  скан того же игрока не теряет игры;
- кэш фич по `cache_key` переиспользует анализ и копирует его в строку текущей
  игры;
- `game_scores` уникальны по `(game_id, model_version)`, re-score идемпотентен;
- зависшие сканы старше `STALE_SCAN_MINUTES` переводятся в `failed` на старте;
- `rated_only` доезжает до Lichess API;
- клиент Lichess отправляет описательный `User-Agent` — без него API отдаёт
  `404` (проверено live-smoke).

**Live smoke (22.10.2026):** 10 blitz-партий `DrNykterstein` → Stockfish 19,
depth 15 / 100 ms → `TrainedScorer` → `completed`, `low_risk 0.3589`
(CI 0.2964–0.4268), 5 evidence. Это проверка пайплайна, а не качества модели.

---

## 7. Что НЕ работает / ограничения

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
- **Live-провайдер LLM не проверен**: ключа пока нет; путь с OpenAI-совместимым
  и Gemini API покрыт тестами и локальным fake-провайдером, но не оплаченным
  внешним API. Без ключа всегда работает deterministic fallback.
- **Фронтенд — demo-UI без аккаунтов**: последние сканы хранятся только в
  `localStorage`, серверной истории и авторизации нет.
- **Нет auth/rate-limit/retention** на публичном API; перед публикацией нужен
  security/privacy review.
- `ScanPipeline` (`workers/pipeline.py`) не используется — прод-путь идёт через
  `ScanWorker`.
- Репозитории-протоколы (`domain/repositories.py`) без реализаций.

---

## 8. Структура репозитория

```
apps/api/app/          FastAPI: роуты, модели БД, воркеры
apps/web/              статический фронтенд (HTML/CSS/ES-модули, без сборки)
packages/
  ingest/              клиент Lichess, PGN, персистентность
  chess_analysis/      Stockfish + агрегация признаков
  feature_engineering/ контракт GameAggregateFeatures
  ml/                  feature contract, data loader, features, train, scorer
  dataset/             старый bounded collector (Lichess Elite/TCEC; не используется)
  llm/                 RAG-объяснение, валидатор, провайдеры, deterministic fallback
scripts/               download_chessfraud.py
reports/               chessfraud_metrics.json
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

## 9. Как запустить

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

### Фронтенд

Статический интерфейс отдаётся тем же процессом на `/app/`: без Node, сборки
и регистрации. Он ходит в тот же `/api/v1` на том же origin, поэтому CORS не
нужен. Последние проверки хранятся только в `localStorage` браузера.

```bash
uvicorn apps.api.app.main:app
# http://127.0.0.1:8000/app/
```

В UI есть форма (ник, 10–50 партий, блиц/рапид, только рейтинговые), живой
прогресс по этапам, вердикт со score/CI/признаками, карточка «Почему такой
вердикт» с бейджем «LLM»/«без LLM», пунктами reasoning и источниками методики,
таблица партий с пагинацией, разбор ходов по партии, светлая/тёмная тема и
недавние сканы.
Ошибки валидации, переполненной очереди, сети, упавшего скана и `404`
показываются текстом с кнопкой повторной попытки.

Score — вероятностный сигнал для дополнительной проверки, **не доказательство
читерства**.

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
python -m pytest -q                          # 229 passed (228 + smoke с артефактом)
python -m ruff check apps packages tests scripts
python -m mypy                               # Success: no issues found in 65 source files
```

Stockfish-тесты пропускаются, если бинарь не найден.

> ⚠️ В репозитории нет `.venv`; используй локальное окружение Python 3.12+
> с `.[dev,dataset,ml]`. Системный Python без зависимостей тесты не запустит.

---

## 10. Бэклог по приоритету

1. **Per-move serving**: протащить вероятность по каждому ходу в
   `GameAggregateFeatures` и `ScoringStage`; оценить, даёт ли это прирост.
2. **Maia-2 на Elo игрока** как human-aligned сигнал (сейчас `maia_divergence`
   пуст).
3. **Больше реальных данных**: ChessFraud-Synth как pretrain → fine-tune на
   `chess_fraud`, внешняя валидация на новых когортах.
4. **Per-game isolation**: одна плохая партия не должна валить scan.
5. **Live-проверка LLM-провайдера** на реальном ключе (Gemini или
   OpenAI-совместимый endpoint) и выбор модели; схема, валидатор и fallback уже
   готовы.
6. **Графики во фронтенде**: CPL/time/top-1 по партиям и экспорт результата;
   базовый UI, polling и разбор ходов уже есть.
7. **Security/privacy**: auth, rate-limit, retention, запрет индексации.
8. **PostgreSQL/Redis** для multi-process worker.

---

## 11. Лицензии и атрибуция

ChessFraud распространяется по CC BY 4.0. Если используешь данные/модель,
цитируй:

```
Linich, A., Lepin, A., Sakhovskiy, A., Toleutaeva, A., Lepa, G.,
Neznamov, A., Budennyy, S. (2026).
ChessFraud: Exploring the Capabilities of Human-Aligned Models for Cheating
Detection in Online Chess. KDD 2026. https://doi.org/10.1145/3770855.3817587
```

ChessFraud-Synth — производное от публичной Lichess DB (CC0).
