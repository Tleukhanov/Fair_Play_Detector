// Methodology view: static, final content. Numbers mirror README and
// reports/chessfraud_metrics.json; update both together.

import { alert, badge, card, el, riskBadge, riskMeter, tableWrap } from "../components.js";

const REPO = "https://github.com/Tleukhanov/Fair_Play_Detector";
const LINKS = {
  readme: `${REPO}/blob/main/README.md`,
  report: `${REPO}/blob/main/reports/chessfraud_metrics.json`,
  contract: `${REPO}/blob/main/packages/ml/feature_contract.py`,
  scorer: `${REPO}/blob/main/packages/ml/scorer.py`,
};

const FEATURES = [
  ["centipawn_loss", "Средняя потеря оценки в сантипешках (POV ходящего)."],
  ["normalized_centipawn_loss", "То же, нормированное на масштаб позиции."],
  ["match_sf1 / sf9 / sf15", "Доля ходов, совпавших с top-1 / top-9 / top-15 Stockfish."],
  ["eval_swing", "Разброс оценки внутри партии."],
  ["abs_eval", "Средний модуль оценки: насколько позиция была острой."],
  ["phase", "Фаза партии, в которой считались агрегаты."],
  ["move_thinking_time", "Среднее время на ход из часов Lichess."],
  ["time_ratio", "Отношение времени хода к остатку на часах."],
  ["player_elo_centered", "Elo игрока, центрированное (только нормализатор)."],
  ["opponent_elo_centered", "Elo соперника, центрированное (только нормализатор)."],
];

const METRICS = [
  ["ROC-AUC", "0.7303", "0.6454–0.8130"],
  ["PR-AUC", "0.7187", "0.6327–0.8270"],
  ["Brier", "0.2005", "0.1755–0.2294"],
  ["Recall @ FPR 5%", "0.1707", "0.1000–0.2746"],
];

export async function renderMethodology(root) {
  root.append(
    el(
      "div",
      { class: "page-head" },
      el("p", { class: "eyebrow" }, "как устроен сигнал"),
      el("h1", {}, "Методология"),
      el(
        "p",
        { class: "lead" },
        "Коротко и проверяемо: какие признаки считаются, как из них получается score, что означают пороги и где заканчивается интерпретация.",
      ),
    ),
    alert(
      "Score — это вероятность машиноподобной игры на данной выборке партий. Он не доказывает читерство и не заменяет ручную проверку.",
      "warn",
      { title: "Главное ограничение" },
    ),
    card(
      {
        title: "Что измеряется",
        subtitle: "12 агрегатов на player-side игру, контракт фич версии 2.0.0.",
        actions: el(
          "a",
          { class: "link-quiet", href: LINKS.contract, target: "_blank", rel: "noopener noreferrer" },
          "feature_contract.py ↗",
        ),
      },
      tableWrap(
        ["Признак", "Смысл"],
        FEATURES.map(([name, meaning]) => [
          el("span", { class: "mono small" }, name),
          meaning,
        ]),
        { caption: "Сырые доски и ходы в модель не попадают — только агрегаты." },
      ),
      el(
        "p",
        { class: "table-note" },
        "Дебютные полуходы (half_move < 21) отсекаются, как в ChessFraud. Elo используется только как нормализатор, чтобы модель не выучила «кто в когорте читерит».",
      ),
    ),
    card(
      {
        title: "Как получается score",
        subtitle: "Game-level модель, затем player-level агрегация.",
      },
      el(
        "ol",
        { class: "list" },
        el("li", {}, "Для каждой player-side игры считаются агрегаты и вероятность подсказки моделью XGBoost (300 деревьев, depth 4, сигмоидная калибровка)."),
        el("li", {}, "Score игрока — среднее вероятностей по играм скана."),
        el("li", {}, "95% интервал — bootstrap по партиям; он показывает неопределённость выборки, а не «точность приговора»."),
        el("li", {}, "Уровень риска выбирается по player-mean порогам: suspicious = 0.4915, high_risk = 0.5638."),
      ),
      el(
        "div",
        { class: "inline-facts", style: { marginTop: "14px" } },
        el("span", {}, "модель: ", el("strong", {}, "chessfraud-v1")),
        el("span", {}, "данные: ", el("strong", {}, "ChessFraud full")),
        el("span", {}, "505 партий · 49 игроков · 1 010 player-games"),
        el("span", {}, "split: player-disjoint 60/20/20"),
      ),
    ),
    card(
      {
        title: "Честные метрики",
        subtitle: "Test: 198 партий, 82 положительных, prevalence 41.4%.",
        actions: el(
          "a",
          { class: "link-quiet", href: LINKS.report, target: "_blank", rel: "noopener noreferrer" },
          "metrics.json ↗",
        ),
      },
      tableWrap(
        [
          "Метрика",
          { label: "Значение", className: "num" },
          { label: "95% CI", className: "num" },
        ],
        METRICS.map(([name, value, interval]) => [
          name,
          el("strong", {}, value),
          el("span", { class: "muted" }, interval),
        ]),
      ),
      el(
        "p",
        { class: "table-note" },
        "Пороги per-game (0.7428 / 0.7764) сохранены только для отчётности: смешивать их с player-mean нельзя.",
      ),
    ),
    card(
      { title: "Как читать уровни" },
      el(
        "div",
        { class: "definition-grid" },
        el(
          "div",
          { class: "definition" },
          el("dt", {}, riskBadge("low_risk")),
          el("dd", {}, "Score ниже 0.49. Признаки в пределах когорты. Это самый частый исход, в том числе для сильных людей."),
        ),
        el(
          "div",
          { class: "definition" },
          el("dt", {}, riskBadge("suspicious")),
          el("dd", {}, "Score 0.49–0.56. Повод посмотреть партии и интервал, а не готовый вывод."),
        ),
        el(
          "div",
          { class: "definition" },
          el("dt", {}, riskBadge("high_risk")),
          el("dd", {}, "Score ≥ 0.56. На текущей когорте почти недостижим (максимум 0.5630 при пороге 0.5638); пороги шумные и требуют пересчёта."),
        ),
      ),
      riskMeter(0.36),
      el(
        "p",
        { class: "table-note" },
        "Метка на шкале показывает пример score 0.36. Пороги построены на validation по 10 игрокам.",
      ),
    ),
    card(
      {
        title: "Ограничения",
        subtitle: "То, что важно прочитать до интерпретации.",
      },
      el(
        "ul",
        { class: "list" },
        el("li", {}, "Маленькая когорта: 49 игроков, тест — 10 игроков / 198 партий. Широкие CI, split чувствителен к seed."),
        el("li", {}, "Prevalence в тесте 41.4% (по датасету 40.3%). Precision с реального трафика (<1%) не переносится; сопоставимая величина — recall при фиксированном FPR."),
        el("li", {}, "Преимущество над простой эвристикой top-1 не доказано: ΔAUC = −0.00013, CI [−0.0598, +0.0556]."),
        el("li", {}, "Per-move модель (OOF ROC-AUC 0.664) обучена, но в serving не подключена; Maia/Allie не участвуют в inference."),
        el("li", {}, "Результат не является доказательством читерства и не основание для автомодерации."),
      ),
    ),
    card(
      {
        title: "Версия и исходники",
        subtitle: "Всё, что можно проверить самостоятельно.",
      },
      el(
        "ul",
        { class: "list" },
        el(
          "li",
          {},
          el("a", { href: LINKS.readme, target: "_blank", rel: "noopener noreferrer" }, "README с метриками и ограничениями"),
        ),
        el(
          "li",
          {},
          el("a", { href: LINKS.report, target: "_blank", rel: "noopener noreferrer" }, "Машиночитаемый отчёт reports/chessfraud_metrics.json"),
        ),
        el(
          "li",
          {},
          el("a", { href: LINKS.scorer, target: "_blank", rel: "noopener noreferrer" }, "Код scorer и пороги"),
        ),
        el(
          "li",
          {},
          el("a", { href: "#/" }, "Запустить проверку"),
        ),
      ),
      el(
        "div",
        { class: "inline-facts", style: { marginTop: "14px" } },
        el("span", {}, badge("chessfraud-v1", "blue")),
        el("span", {}, "контракт фич 2.0.0"),
        el("span", {}, "Stockfish 17 · depth 15"),
      ),
    ),
  );
}
