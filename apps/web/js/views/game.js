// Game view: static moves preview. The Result/Games wave replaces the demo
// rows with GET /scans/{id}/games/{gameId} (played_move_uci, CPL, top-N flags).

import { alert, badge, card, el, tableWrap } from "../components.js";

const DEMO_MOVES = [
  { ply: 1, move: "e4", cpl: 12, top1: true, top3: true, time: 4.2 },
  { ply: 2, move: "e5", cpl: 18, top1: false, top3: true, time: 3.1 },
  { ply: 3, move: "Nf3", cpl: 9, top1: true, top3: true, time: 5.0 },
  { ply: 4, move: "Nc6", cpl: 26, top1: false, top3: false, time: 2.4 },
  { ply: 5, move: "Bb5", cpl: 7, top1: true, top3: true, time: 6.3 },
  { ply: 6, move: "a6", cpl: 41, top1: false, top3: false, time: 1.8 },
  { ply: 7, move: "Ba4", cpl: 14, top1: true, top3: true, time: 8.1 },
  { ply: 8, move: "Nf6", cpl: 22, top1: false, top3: true, time: 5.7 },
  { ply: 9, move: "O-O", cpl: 11, top1: true, top3: true, time: 7.4 },
  { ply: 10, move: "Be7", cpl: 33, top1: false, top3: true, time: 3.9 },
  { ply: 11, move: "Re1", cpl: 16, top1: false, top3: true, time: 9.2 },
  { ply: 12, move: "b5", cpl: 48, top1: false, top3: false, time: 2.2 },
];

function cplClass(cpl) {
  if (cpl <= 10) return "cpl-low";
  if (cpl <= 25) return "cpl-mid";
  return "cpl-high";
}

export async function renderGame(root, params = {}) {
  const scanId = params.scanId || "demo";
  const gameId = params.gameId || "demo-game-1";
  const resultHref = `#/scan/${encodeURIComponent(scanId)}/result`;

  const headers = [
    { label: "Пли", className: "num" },
    "Ход",
    { label: "CPL", className: "num" },
    { label: "Top-1", className: "center" },
    { label: "Top-3", className: "center" },
    { label: "Время", className: "num" },
  ];

  const rows = DEMO_MOVES.map((move) => [
    String(move.ply),
    el("span", {}, move.move),
    el("span", { class: cplClass(move.cpl) }, String(move.cpl)),
    move.top1 ? badge("да", "good") : el("span", { class: "muted" }, "—"),
    move.top3 ? badge("да", "neutral") : el("span", { class: "muted" }, "—"),
    `${move.time.toFixed(1)} с`,
  ]);

  root.append(
    el(
      "nav",
      { class: "breadcrumbs", "aria-label": "Навигация" },
      el("a", { href: "#/" }, "Проверка"),
      el("span", { class: "sep", "aria-hidden": "true" }, "/"),
      el("a", { href: resultHref }, "Вердикт"),
      el("span", { class: "sep", "aria-hidden": "true" }, "/"),
      el("span", {}, "Партия"),
    ),
    el(
      "div",
      { class: "page-head page-head-row" },
      el(
        "div",
        {},
        el("p", { class: "eyebrow mono" }, gameId),
        el("h1", {}, "Разбор партии"),
        el(
          "p",
          { class: "lead" },
          "Ходы, потеря оценки и совпадения с лучшими линиями Stockfish для одной партии из скана.",
        ),
      ),
      el(
        "div",
        { class: "head-actions" },
        badge("демо · предпросмотр", "demo"),
        el("a", { class: "btn btn-secondary btn-sm", href: resultHref }, "К вердикту"),
      ),
    ),
    alert(
      "Ходы и метрики ниже — демонстрационные. Реальные строки подставит следующий этап.",
      "info",
      { title: "Предпросмотр интерфейса" },
    ),
    card(
      { title: "Партия", subtitle: "Метаданные из публичного API Lichess." },
      el(
        "div",
        { class: "metric-grid" },
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, "1–0"),
          el("div", { class: "metric-label" }, "результат"),
          el("div", { class: "metric-note" }, "белые"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, "68"),
          el("div", { class: "metric-label" }, "полуходов"),
          el("div", { class: "metric-note" }, "проанализировано 12"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, "blitz"),
          el("div", { class: "metric-label" }, "контроль"),
          el("div", { class: "metric-note" }, "rated"),
        ),
      ),
      el(
        "p",
        { class: "table-note" },
        "CPL — centipawn loss в POV ходящего; значение ограничено клипом мата.",
      ),
    ),
    card(
      {
        title: "Ходы",
        subtitle: "Наведите на строку, чтобы выделить ход. Таблица прокручивается на узких экранах.",
        actions: el("span", { class: "badge badge-neutral" }, "демо"),
      },
      tableWrap(headers, rows, {
        className: "moves-table",
        caption: "Первые 12 полуходов · демонстрационные значения",
      }),
    ),
    el(
      "div",
      { class: "card-footer", style: { borderTop: "0", paddingTop: "0", marginTop: "0" } },
      el("a", { class: "btn btn-secondary", href: resultHref }, "Назад к вердикту"),
      el("a", { class: "btn btn-primary", href: "#/" }, "Новая проверка"),
    ),
  );
}
