// Scan view: static progress shell. The Scan Flow wave replaces the demo
// numbers with polling of GET /api/v1/scans/{id} and redirects on completion.

import { badge, card, el, progress, skeletonRows, stepper } from "../components.js";

export async function renderScan(root, params = {}) {
  const scanId = params.scanId || "demo";
  const shortId = scanId.length > 12 ? `${scanId.slice(0, 8)}…` : scanId;

  root.append(
    el(
      "nav",
      { class: "breadcrumbs", "aria-label": "Навигация" },
      el("a", { href: "#/" }, "Проверка"),
      el("span", { class: "sep", "aria-hidden": "true" }, "/"),
      el("span", {}, "Сканирование"),
    ),
    el(
      "div",
      { class: "page-head page-head-row" },
      el(
        "div",
        {},
        el("p", { class: "eyebrow" }, `скан ${shortId}`),
        el("h1", {}, "Проверка выполняется"),
        el(
          "p",
          { class: "lead" },
          "Идёт загрузка публичных партий и анализ Stockfish. Эта страница обновляется автоматически после подключения polling.",
        ),
      ),
      el(
        "div",
        { class: "head-actions" },
        badge("демо", "neutral"),
        el("a", { class: "btn btn-secondary btn-sm", href: "#/" }, "Новая проверка"),
      ),
    ),
    card(
      {
        title: "Прогресс",
        subtitle: "Статический предпросмотр — живые данные появятся в следующем коммите.",
        actions: badge("analyzing", "blue"),
      },
      progress(62, {
        label: "Анализ партий",
        detail: "19 из 30",
        metaLeft: "Stockfish · depth 15 · 100 ms",
        metaRight: "≈ 1 мин осталось",
      }),
      el(
        "div",
        { class: "metric-grid", style: { marginTop: "16px" } },
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, "30"),
          el("div", { class: "metric-label" }, "найдено партий"),
          el("div", { class: "metric-note" }, "blitz · rated"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, "19"),
          el("div", { class: "metric-label" }, "проанализировано"),
          el("div", { class: "metric-note" }, "без ошибок"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, "11"),
          el("div", { class: "metric-label" }, "в очереди"),
          el("div", { class: "metric-note" }, "≈ 1 мин"),
        ),
      ),
    ),
    card(
      { title: "Этапы", subtitle: "Пайплайн скана: от загрузки PGN до вердикта." },
      stepper(
        [
          { label: "В очереди", detail: "queued" },
          { label: "Загрузка партий", detail: "fetching · 30" },
          { label: "Анализ Stockfish", detail: "analyzing · 19/30" },
          { label: "Скоринг", detail: "scoring" },
          { label: "Объяснение", detail: "explaining" },
        ],
        2,
      ),
    ),
    card(
      {
        title: "Предпросмотр результатов",
        subtitle: "Плейсхолдеры, пока идёт анализ. Реальные строки появятся после этапа scoring.",
      },
      skeletonRows(5, 4),
      el(
        "p",
        { class: "skeleton-caption" },
        "Таблица партий и вердикт откроются автоматически после завершения скана.",
      ),
    ),
  );
}
