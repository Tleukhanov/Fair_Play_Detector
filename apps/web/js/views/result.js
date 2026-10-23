// Result view: static verdict preview with clearly labelled demo data.
// The Result wave replaces the sample values with GET /scans/{id}/verdict.

import {
  alert,
  badge,
  card,
  el,
  evidenceRow,
  riskBadge,
  riskMeter,
} from "../components.js";

const DEMO_SCORE = 0.36;
const DEMO_EVIDENCE = [
  { code: "top_1_match_rate", label: "Совпадение с top-1", value: 0.62, baseline: 0.55, weight: 0.28, unit: "%" },
  { code: "match_sf9", label: "Совпадение с top-9", value: 0.71, baseline: 0.66, weight: 0.18, unit: "%" },
  { code: "mean_centipawn_loss", label: "Средний CPL", value: 24, baseline: 31, weight: 0.22, unit: "cp" },
  { code: "move_thinking_time", label: "Время на ход", value: 7.2, baseline: 6.4, weight: 0.16, unit: "s" },
  { code: "eval_swing", label: "Разброс оценки", value: 0.18, baseline: 0.21, weight: 0.16 },
];

const DEMO_LIMITATIONS = [
  "Когорта обучения маленькая: 49 игроков, тест — 10 игроков / 198 партий. Интервалы широкие.",
  "Prevalence в ChessFraud — 40.3%; precision с реального трафика (<1%) не переносится.",
  "Модель не показала доказанного превосходства над эвристикой top-1 на этой выборке.",
  "Результат не является доказательством читерства и не предназначен для автомодерации.",
];

export async function renderResult(root, params = {}) {
  const scanId = params.scanId || "demo";
  const backHref = `#/scan/${encodeURIComponent(scanId)}`;
  const gameHref = `#/scan/${encodeURIComponent(scanId)}/game/demo-game-1`;

  root.append(
    el(
      "nav",
      { class: "breadcrumbs", "aria-label": "Навигация" },
      el("a", { href: "#/" }, "Проверка"),
      el("span", { class: "sep", "aria-hidden": "true" }, "/"),
      el("a", { href: backHref }, "Скан"),
      el("span", { class: "sep", "aria-hidden": "true" }, "/"),
      el("span", {}, "Вердикт"),
    ),
    el(
      "div",
      { class: "page-head page-head-row" },
      el(
        "div",
        {},
        el("p", { class: "eyebrow" }, "player-level · chessfraud-v1"),
        el("h1", {}, "Результат проверки"),
        el(
          "p",
          { class: "lead" },
          "Средняя вероятность машиноподобной игры по проанализированным партиям и 95% bootstrap-интервал.",
        ),
      ),
      el(
        "div",
        { class: "head-actions" },
        badge("демо · предпросмотр", "demo"),
        riskBadge("low_risk"),
      ),
    ),
    alert(
      "Это демонстрационный макет с вымышленными числами. Он показывает структуру вердикта; реальные значения подставит следующий этап.",
      "warn",
      { title: "Предпросмотр интерфейса" },
    ),
    card(
      {
        className: "verdict-card",
        actions: badge("демо", "demo"),
      },
      el(
        "div",
        { class: "verdict-top" },
        el(
          "div",
          { class: "verdict-user" },
          el("h2", {}, "DrNykterstein"),
          riskBadge("low_risk"),
          el(
            "p",
            { class: "verdict-meta" },
            "30 партий · blitz · только рейтинговые · модель chessfraud-v1",
          ),
        ),
        el("span", { class: "badge badge-neutral" }, "завершено"),
      ),
      el(
        "div",
        { class: "metric-grid" },
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, DEMO_SCORE.toFixed(2)),
          el("div", { class: "metric-label" }, "риск, 0–1"),
          el("div", { class: "metric-note" }, "среднее по партиям"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value mono" }, "0.30–0.43"),
          el("div", { class: "metric-label" }, "95% интервал"),
          el("div", { class: "metric-note" }, "bootstrap по партиям"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, "30"),
          el("div", { class: "metric-label" }, "партий в выборке"),
          el("div", { class: "metric-note" }, "10–50 допустимо"),
        ),
      ),
      riskMeter(DEMO_SCORE),
      el(
        "p",
        { class: "risk-summary" },
        "Профиль ближе к низкому риску: совпадения ходов и потеря оценки не выходят за пределы когорты. ",
        "Интервал и ограничения ниже важнее одного числа — трактовать результат как обвинение нельзя.",
      ),
    ),
    card(
      {
        title: "Признаки",
        subtitle: "Вклад признаков в решение. База — среднее по когорте ChessFraud.",
        actions: el("span", { class: "badge badge-neutral" }, "5 признаков"),
      },
      el(
        "div",
        { class: "evidence-list" },
        DEMO_EVIDENCE.map((item) => evidenceRow(item)),
      ),
      el(
        "p",
        { class: "table-note" },
        "Полоса — наблюдение, вертикальная метка — база. Демо-значения не являются результатом реального скана.",
      ),
    ),
    card(
      {
        title: "Ограничения",
        subtitle: "Что нельзя выводить из этого результата.",
      },
      el(
        "ul",
        { class: "list" },
        DEMO_LIMITATIONS.map((text) => el("li", {}, text)),
      ),
    ),
    el(
      "div",
      { class: "card-footer", style: { borderTop: "0", paddingTop: "0", marginTop: "0" } },
      el("a", { class: "btn btn-secondary", href: backHref }, "К прогрессу скана"),
      el(
        "div",
        { class: "head-actions" },
        el("a", { class: "btn btn-ghost", href: gameHref }, "Демо-партия"),
        el("a", { class: "btn btn-primary", href: "#/" }, "Новая проверка"),
      ),
    ),
  );
}
