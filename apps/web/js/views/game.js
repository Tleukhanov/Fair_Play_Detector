// Game view: metadata and analyzed moves for one game of a scan.
//
// Data comes from GET /scans/{id}/games/{gameId}. Every untrusted string is
// appended through text nodes; the only HTML attributes are validated URLs.

import { ApiError, getGame } from "../api.js";
import { alert, card, el, empty, skeletonRows, tableWrap } from "../components.js";
import {
  formatCpl,
  formatDateTimeUtc,
  formatPercent,
  formatResult,
  formatScore,
} from "../format.js";

const CPL_LOW_MAX = 30;
const CPL_MID_MAX = 80;

function safeHttpUrl(value) {
  if (typeof value !== "string" || value.trim() === "") return null;
  try {
    const url = new URL(value, window.location.href);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    return url.href;
  } catch {
    return null;
  }
}

function cplClass(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (number <= CPL_LOW_MAX) return "cpl-low";
  if (number <= CPL_MID_MAX) return "cpl-mid";
  return "cpl-high";
}

function colorLabel(color) {
  if (color === "white") return "белые";
  if (color === "black") return "чёрные";
  return typeof color === "string" && color !== "" ? color : "—";
}

function topCell(move) {
  const parts = [];
  if (move?.is_top_1) parts.push("1");
  if (move?.is_top_3) parts.push("3");
  return parts.length ? parts.join(" · ") : "—";
}

function sanCell(move) {
  const san =
    typeof move?.played_move_san === "string" && move.played_move_san.trim() !== ""
      ? move.played_move_san
      : null;
  const uci =
    typeof move?.played_move_uci === "string" && move.played_move_uci.trim() !== ""
      ? move.played_move_uci
      : null;
  const label = san || uci || "—";
  return el("span", { class: "mono" }, label);
}

function timeCell(value) {
  const number = Number(value);
  if (value === null || value === undefined || !Number.isFinite(number)) return "—";
  return `${number.toFixed(1)} с`;
}

function movesTable(moves) {
  const headers = [
    { label: "Пли", className: "num" },
    "SAN",
    { label: "CPL", className: "num" },
    { label: "Top-1 / Top-3", className: "center" },
    { label: "Время", className: "num" },
    { label: "Maia", className: "num" },
  ];
  const rows = moves.map((move) => [
    move?.ply === null || move?.ply === undefined ? "—" : String(move.ply),
    sanCell(move),
    el(
      "span",
      { class: cplClass(move?.centipawn_loss) },
      formatCpl(move?.centipawn_loss),
    ),
    topCell(move),
    timeCell(move?.time_seconds),
    move?.maia_probability === null || move?.maia_probability === undefined
      ? "—"
      : formatPercent(move.maia_probability, 0),
  ]);
  const wrap = tableWrap(headers, rows, {
    className: "moves-table",
    caption: `${moves.length} полуходов · CPL в сантипешках, POV ходящего`,
  });
  wrap.classList.add("moves-scroll");
  return wrap;
}

function metaCard(game) {
  const source = safeHttpUrl(game?.source_url);
  const moveCount = Number(game?.move_count);
  const maia = Number(game?.maia_divergence);
  const facts = [
    ["дата", formatDateTimeUtc(game?.played_at)],
    ["контроль", game?.speed || "—"],
    ["цвет", colorLabel(game?.user_color)],
    ["результат", typeof game?.result === "string" ? game.result : "—"],
    ["Maia", game?.maia_divergence === null || game?.maia_divergence === undefined ? "—" : formatScore(maia)],
  ];

  return card(
    {
      title: "Партия",
      subtitle: "Метаданные из публичного API Lichess и сохранённого анализа.",
      actions: source
        ? el(
            "a",
            {
              class: "btn btn-secondary btn-sm",
              href: source,
              target: "_blank",
              rel: "noopener noreferrer",
            },
            "Открыть на Lichess ↗",
          )
        : null,
    },
    el(
      "div",
      { class: "metric-grid" },
      el(
        "div",
        { class: "metric" },
        el("div", { class: "metric-value" }, formatResult(game?.result, game?.user_color)),
        el("div", { class: "metric-label" }, "результат"),
        el("div", { class: "metric-note" }, game?.result || "—"),
      ),
      el(
        "div",
        { class: "metric" },
        el("div", { class: "metric-value" }, Number.isFinite(moveCount) ? String(moveCount) : "—"),
        el("div", { class: "metric-label" }, "полуходов"),
        el("div", { class: "metric-note" }, "проанализировано"),
      ),
      el(
        "div",
        { class: "metric" },
        el("div", { class: "metric-value" }, formatPercent(game?.top_move_match_rate, 0)),
        el("div", { class: "metric-label" }, "top-1"),
        el("div", { class: "metric-note" }, "совпадений с Stockfish"),
      ),
      el(
        "div",
        { class: "metric" },
        el("div", { class: "metric-value" }, formatCpl(game?.mean_centipawn_loss)),
        el("div", { class: "metric-label" }, "средний CPL"),
        el("div", { class: "metric-note" }, "POV ходящего"),
      ),
    ),
    el(
      "div",
      { class: "inline-facts", style: { marginTop: "14px" } },
      facts.map(([label, value]) => el("span", {}, `${label}: `, el("strong", {}, value))),
    ),
    el(
      "p",
      { class: "table-note" },
      "CPL — centipawn loss в POV ходящего, значение ограничено клипом мата. " +
        "Maia/Allie не подключены к inference, поэтому maia_divergence обычно пуст.",
    ),
  );
}

function movesCard(moves) {
  if (moves.length === 0) {
    return card(
      { title: "Ходы" },
      empty({
        icon: null,
        title: "Ходы не сохранены",
        text: "Для этой партии backend не вернул проанализированные полуходы.",
      }),
    );
  }
  return card(
    {
      title: "Ходы",
      subtitle:
        "Таблица прокручивается на узких экранах. 1 — ход совпал с top-1 Stockfish, 3 — попал в top-3.",
    },
    movesTable(moves),
    el(
      "p",
      { class: "table-note" },
      "Зелёный CPL ≤ 30, жёлтый — 31–80, красный — больше 80. Время «—» означает, что часов в PGN не было.",
    ),
  );
}

function errorText(error) {
  const status = Number(error?.status) || 0;
  if (status === 0) return "Сервер недоступен. Проверьте подключение и повторите.";
  if (status >= 500) return `Сервер вернул ошибку ${status}. Попробуйте ещё раз позже.`;
  if (error instanceof Error && error.message && /[А-Яа-яЁё]/.test(error.message)) {
    return error.message;
  }
  return "Не удалось загрузить партию. Попробуйте ещё раз.";
}

function errorCard(title, error, retry) {
  return card(
    { title },
    alert(errorText(error), "error", { title: "Ошибка загрузки" }),
    el(
      "div",
      { class: "form-actions" },
      el(
        "button",
        { type: "button", class: "btn btn-secondary", onclick: () => retry() },
        "Повторить",
      ),
      el("a", { class: "btn btn-ghost", href: "#/" }, "На главную"),
    ),
  );
}

function notFoundCard(scanId, scanMissing = false) {
  const title = scanMissing ? "Скан не найден" : "Партия не найдена";
  return card(
    { title },
    alert(
      scanMissing
        ? "Сервер не знает скан с таким идентификатором. Возможно, ссылка устарела."
        : "Backend не нашёл эту партию в указанном скане. Возможно, ссылка устарела.",
      scanMissing ? "error" : "warn",
      { title },
    ),
    el(
      "div",
      { class: "form-actions" },
      scanMissing
        ? el("a", { class: "btn btn-primary", href: "#/" }, "На главную")
        : el(
            "a",
            { class: "btn btn-secondary", href: `#/scan/${encodeURIComponent(scanId)}/result` },
            "Назад к вердикту",
          ),
    ),
  );
}

export async function renderGame(root, params = {}) {
  const scanId = params.scanId;
  const gameId = params.gameId;
  if (typeof scanId !== "string" || scanId === "" || typeof gameId !== "string" || gameId === "") {
    root.append(
      alert("В адресе нет идентификатора скана или партии.", "error", { title: "Неверная ссылка" }),
    );
    return;
  }

  const resultHref = `#/scan/${encodeURIComponent(scanId)}/result`;
  const eyebrow = el("p", { class: "eyebrow mono" }, gameId);
  const slot = el("div", { class: "game-slot" });

  root.append(
    el(
      "nav",
      { class: "breadcrumbs", "aria-label": "Навигация" },
      el("a", { href: "#/" }, "Проверка"),
      el("span", { class: "sep", "aria-hidden": "true" }, "/"),
      el("a", { href: `#/scan/${encodeURIComponent(scanId)}` }, "Скан"),
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
        eyebrow,
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
        el("a", { class: "btn btn-secondary btn-sm", href: resultHref }, "К вердикту"),
      ),
    ),
    slot,
    el(
      "div",
      { class: "card-footer", style: { borderTop: "0", paddingTop: "0", marginTop: "0" } },
      el("a", { class: "btn btn-secondary", href: resultHref }, "Назад к вердикту"),
      el("a", { class: "btn btn-primary", href: "#/" }, "Новая проверка"),
    ),
  );

  function skeleton() {
    return [
      card(
        { title: "Партия", subtitle: "Загружаем метаданные…" },
        skeletonRows(3, 3),
      ),
      card(
        { title: "Ходы", subtitle: "Загружаем разбор…" },
        skeletonRows(8, 4),
        el("p", { class: "skeleton-caption" }, "Таблица полуходов появится здесь."),
      ),
    ];
  }

  async function load() {
    slot.setAttribute("aria-busy", "true");
    slot.replaceChildren(...skeleton());
    try {
      const data = await getGame(scanId, gameId);
      if (!slot.isConnected) return;
      const game = data?.game && typeof data.game === "object" ? data.game : {};
      const moves = Array.isArray(data?.moves) ? data.moves : [];
      if (typeof game.game_id === "string" && game.game_id !== "") {
        eyebrow.textContent = `партия · ${game.game_id}`;
      }
      slot.replaceChildren(metaCard(game), movesCard(moves));
    } catch (error) {
      if (!slot.isConnected) return;
      if (error instanceof ApiError && error.status === 404) {
        const detail = typeof error.detail === "string" ? error.detail : "";
        slot.replaceChildren(notFoundCard(scanId, /scan not found/i.test(detail)));
        return;
      }
      slot.replaceChildren(errorCard("Не удалось загрузить партию", error, load));
    } finally {
      if (slot.isConnected) slot.setAttribute("aria-busy", "false");
    }
  }

  await load();
}
