// Result view: real player-level verdict plus the paginated games table.
//
// Data comes from GET /scans/{id}/verdict and GET /scans/{id}/games. All API
// values are rendered as text nodes through components.js; raw HTML is never
// injected.

import { ApiError, getGames, getVerdict } from "../api.js";
import {
  alert,
  badge,
  card,
  el,
  empty,
  pagination,
  riskMeter,
  skeletonRows,
  tableWrap,
} from "../components.js";
import {
  evidenceLabel,
  formatCi,
  formatCpl,
  formatDateTimeUtc,
  formatEvidenceBaseline,
  formatEvidenceValue,
  formatPercent,
  formatResult,
  formatScore,
  formatWeight,
  riskClass,
  riskLabel,
} from "../format.js";

const PAGE_SIZE = 20;

const STATIC_LIMITATIONS = [
  "Это вероятностный сигнал, а не доказательство читерства и не основание для автомодерации.",
  "Когорта маленькая: 49 игроков, тест — 10 игроков / 198 партий; интервалы широкие.",
  "Prevalence в тесте 41.4%, на реальном трафике — меньше 1%; precision не переносится.",
  "Maia/Allie не подключены к inference, поэтому maia_divergence обычно пуст.",
  "Преимущество над простой эвристикой top-1 на этой когорте не доказано.",
];

const RISK_WORDS = {
  low_risk: "низкий",
  suspicious: "подозрительный",
  high_risk: "высокий",
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function numberOrNull(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function gameHash(scanId, gameId) {
  return `#/scan/${encodeURIComponent(scanId)}/game/${encodeURIComponent(gameId)}`;
}

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

function pageHeader(scanId) {
  const eyebrow = el("p", { class: "eyebrow" }, "player-level");
  return {
    eyebrow,
    node: el(
      "div",
      { class: "page-head page-head-row" },
      el(
        "div",
        {},
        eyebrow,
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
        el(
          "a",
          { class: "btn btn-secondary btn-sm", href: `#/scan/${encodeURIComponent(scanId)}` },
          "К прогрессу скана",
        ),
      ),
    ),
  };
}

function breadcrumbs(scanId) {
  return el(
    "nav",
    { class: "breadcrumbs", "aria-label": "Навигация" },
    el("a", { href: "#/" }, "Проверка"),
    el("span", { class: "sep", "aria-hidden": "true" }, "/"),
    el("a", { href: `#/scan/${encodeURIComponent(scanId)}` }, "Скан"),
    el("span", { class: "sep", "aria-hidden": "true" }, "/"),
    el("span", {}, "Вердикт"),
  );
}

function verdictSkeleton() {
  return card(
    { title: "Вердикт", subtitle: "Загружаем результат скана…" },
    skeletonRows(3, 3),
    el("p", { class: "skeleton-caption" }, "Score, интервал и признаки появятся здесь."),
  );
}

function gamesSkeleton() {
  return card(
    { title: "Партии", subtitle: "Загружаем список партий…" },
    skeletonRows(6, 4),
    el("p", { class: "skeleton-caption" }, "Таблица партий и пагинация появятся здесь."),
  );
}

function errorText(error, fallback) {
  const status = Number(error?.status) || 0;
  if (status === 0) {
    return "Сервер недоступен. Проверьте подключение и повторите.";
  }
  if (status === 404 || status === 410) {
    return "Данные не найдены. Возможно, ссылка устарела или скан был удалён.";
  }
  if (status >= 500) {
    return `Сервер вернул ошибку ${status}. Попробуйте ещё раз позже.`;
  }
  if (error instanceof Error && error.message && /[А-Яа-яЁё]/.test(error.message)) {
    return error.message;
  }
  return fallback;
}

function errorCard(title, error, retry) {
  return card(
    { title },
    alert(errorText(error, "Не удалось загрузить данные. Попробуйте ещё раз."), "error", {
      title: "Ошибка загрузки",
    }),
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

function missingScanCard() {
  return card(
    { title: "Скан не найден" },
    alert("Сервер не знает скан с таким идентификатором. Возможно, ссылка устарела.", "error", {
      title: "Скан не найден",
    }),
    el(
      "div",
      { class: "form-actions" },
      el("a", { class: "btn btn-primary", href: "#/" }, "На главную"),
    ),
  );
}

function pendingCard(scanId) {
  return card(
    { title: "Вердикт ещё не готов" },
    alert(
      "Скан ещё выполняется или вердикт пока недоступен. Открываем страницу скана…",
      "info",
      { title: "Вердикт ещё не готов" },
    ),
    el(
      "div",
      { class: "form-actions" },
      el(
        "a",
        { class: "btn btn-secondary btn-sm", href: `#/scan/${encodeURIComponent(scanId)}` },
        "К скану",
      ),
    ),
  );
}

function evidenceRow(item) {
  const code = typeof item?.code === "string" ? item.code : "";
  const value = numberOrNull(item?.value);
  const baseline = numberOrNull(item?.baseline);
  const scale = Math.max(Math.abs(value || 0), Math.abs(baseline || 0), 1e-6);
  const fillPercent = value === null ? 0 : clamp((Math.abs(value) / scale) * 100, 0, 100);
  const baselinePercent =
    baseline === null ? 0 : clamp((Math.abs(baseline) / scale) * 100, 0, 100);
  return el(
    "div",
    { class: "evidence-row" },
    el(
      "div",
      { class: "evidence-main" },
      el(
        "div",
        { class: "evidence-head" },
        el(
          "span",
          { class: "evidence-label" },
          evidenceLabel(code, item?.label),
          code ? el("span", { class: "evidence-code mono" }, code) : null,
        ),
        el("span", { class: "evidence-weight" }, `вес ${formatWeight(item?.weight)}`),
      ),
      el(
        "div",
        { class: "evidence-bar", "aria-hidden": "true" },
        el("span", { class: "evidence-fill", style: { width: `${fillPercent}%` } }),
        el("span", { class: "evidence-baseline", style: { left: `${baselinePercent}%` } }),
      ),
    ),
    el(
      "div",
      { class: "evidence-values" },
      el("span", { class: "evidence-value" }, formatEvidenceValue(code, value)),
      el("span", { class: "evidence-base" }, `база ${formatEvidenceBaseline(code, baseline)}`),
    ),
  );
}

function russianSummary(verdict) {
  const level = typeof verdict?.risk_level === "string" ? verdict.risk_level : "";
  const word = RISK_WORDS[level] || riskLabel(level).toLowerCase();
  const score = formatScore(verdict?.risk_score);
  const interval = formatCi(verdict?.confidence_interval);
  const games = numberOrNull(verdict?.games_analyzed);
  const parts = [level ? `Риск: ${word}.` : "Уровень риска не определён."];
  if (score !== "—") {
    parts.push(
      interval === "—" ? `Score ${score}.` : `Score ${score}, 95% ДИ ${interval}.`,
    );
  } else {
    parts.push("Score не рассчитан.");
  }
  if (games !== null) {
    parts.push(`Сигнал основан на ${Math.round(games)} партиях.`);
  }
  if (verdict?.risk_level === "suspicious" || verdict?.risk_level === "high_risk") {
    parts.push("Стоит посмотреть отдельные партии и ширину интервала — это не приговор.");
  } else {
    parts.push("Это вероятностный сигнал, а не доказательство.");
  }
  return parts.join(" ");
}

function verdictCard(verdict) {
  const evidence = Array.isArray(verdict?.evidence) ? verdict.evidence : [];
  const rawLimitations = Array.isArray(verdict?.limitations)
    ? verdict.limitations.filter((item) => typeof item === "string" && item.trim() !== "")
    : [];
  const games = numberOrNull(verdict?.games_analyzed);
  const score = numberOrNull(verdict?.risk_score);

  const evidenceCard = card(
    {
      title: "Признаки",
      subtitle: "Вклад признаков в решение. База — среднее по когорте ChessFraud.",
      actions: badge(`${evidence.length} признаков`, "neutral"),
    },
    evidence.length
      ? el("div", { class: "evidence-list" }, evidence.map((item) => evidenceRow(item)))
      : empty({
          icon: null,
          title: "Признаки недоступны",
          text: "Для этого вердикта backend не вернул объяснений.",
        }),
    el(
      "p",
      { class: "table-note" },
      "Полоса — наблюдение, вертикальная метка — база когорты. Ширина полосы относительна максимуму из двух значений.",
    ),
  );

  const limitationsCard = card(
    { title: "Ограничения", subtitle: "Что нельзя выводить из этого результата." },
    el(
      "ul",
      { class: "list" },
      STATIC_LIMITATIONS.map((text) => el("li", {}, text)),
    ),
    rawLimitations.length
      ? el(
          "details",
          { class: "raw-details" },
          el("summary", {}, "Ограничения из backend (EN)"),
          el(
            "ul",
            { class: "list small details-body" },
            rawLimitations.map((text) => el("li", {}, text)),
          ),
        )
      : null,
  );

  return [
    card(
      {
        className: "verdict-card",
        actions: badge("завершено", "neutral"),
      },
      el(
        "div",
        { class: "verdict-top" },
        el(
          "div",
          { class: "verdict-user" },
          el("h2", {}, verdict?.username || "Игрок"),
          badge(riskLabel(verdict?.risk_level), riskClass(verdict?.risk_level)),
          el(
            "p",
            { class: "verdict-meta" },
            [
              games !== null ? `${Math.round(games)} партий` : null,
              verdict?.model_version ? `модель ${verdict.model_version}` : null,
            ]
              .filter(Boolean)
              .join(" · ") || "player-level",
          ),
        ),
      ),
      el(
        "div",
        { class: "metric-grid" },
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, formatScore(score)),
          el("div", { class: "metric-label" }, "риск, 0–1"),
          el("div", { class: "metric-note" }, "среднее по партиям"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value mono" }, formatCi(verdict?.confidence_interval)),
          el("div", { class: "metric-label" }, "95% интервал"),
          el("div", { class: "metric-note" }, "bootstrap по партиям"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, games !== null ? String(Math.round(games)) : "—"),
          el("div", { class: "metric-label" }, "партий в выборке"),
          el("div", { class: "metric-note" }, "минимум 10"),
        ),
        el(
          "div",
          { class: "metric" },
          el("div", { class: "metric-value" }, el("span", { class: "small" }, verdict?.model_version || "—")),
          el("div", { class: "metric-label" }, "модель"),
          el("div", { class: "metric-note" }, "game-level"),
        ),
      ),
      score !== null
        ? riskMeter(score)
        : el(
            "p",
            { class: "table-note" },
            "Шкала риска недоступна: backend не вернул score для этого скана.",
          ),
      el("p", { class: "risk-summary" }, russianSummary(verdict)),
      el(
        "details",
        { class: "raw-details" },
        el("summary", {}, "Технический вывод модели (EN)"),
        el(
          "div",
          { class: "details-body" },
          el("p", { class: "small" }, verdict?.summary || "—"),
        ),
      ),
    ),
    evidenceCard,
    limitationsCard,
  ];
}

function resultCell(value, userColor) {
  const label = formatResult(value, userColor);
  return label;
}

function sourceCell(sourceUrl) {
  const href = safeHttpUrl(sourceUrl);
  if (!href) return el("span", { class: "muted" }, "—");
  return el(
    "a",
    {
      href,
      target: "_blank",
      rel: "noopener noreferrer",
      class: "link-quiet nowrap",
    },
    "Lichess ↗",
  );
}

function gamesTable(items, scanId, total, page) {
  const headers = [
    { label: "Дата", className: "nowrap" },
    "Результат",
    "Контроль",
    { label: "Ходы", className: "num" },
    { label: "Top-1", className: "num" },
    { label: "CPL", className: "num" },
    { label: "Maia", className: "num" },
    "Lichess",
  ];
  const rows = items.map((item) => [
    formatDateTimeUtc(item?.played_at),
    formatResult(item?.result, item?.user_color),
    item?.speed || "—",
    el("span", {}, numberOrNull(item?.move_count) !== null ? String(item.move_count) : "—"),
    formatPercent(item?.top_move_match_rate, 0),
    formatCpl(item?.mean_centipawn_loss),
    numberOrNull(item?.maia_divergence) === null ? "—" : formatScore(item?.maia_divergence),
    sourceCell(item?.source_url),
  ]);

  const wrap = tableWrap(headers, rows, {
    className: "games-table",
    caption: `Страница ${page} · всего ${total} партий`,
  });
  const table = wrap.querySelector("table");
  if (table) {
    const bodyRows = Array.from(table.tBodies[0]?.rows || []);
    bodyRows.forEach((row, index) => {
      const item = items[index];
      if (!item || typeof item.game_id !== "string" || item.game_id === "") return;
      const href = gameHash(scanId, item.game_id);
      row.tabIndex = 0;
      row.style.cursor = "pointer";
      row.setAttribute("aria-label", `Открыть разбор партии ${item.game_id}`);
      row.addEventListener("click", (event) => {
        if (event.target instanceof Element && event.target.closest("a, button")) return;
        window.location.hash = href;
      });
      row.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        window.location.hash = href;
      });
    });
  }
  return wrap;
}

function gamesCard(data, scanId, requestPage, loadPage) {
  const items = Array.isArray(data?.items) ? data.items : [];
  const total = numberOrNull(data?.total) ?? items.length;
  const page = numberOrNull(data?.page) ?? requestPage;
  const pageSize = numberOrNull(data?.page_size) ?? PAGE_SIZE;

  if (items.length === 0) {
    return card(
      { title: "Партии", actions: badge("0", "neutral") },
      empty({
        title: "Партий нет",
        text: "В этом скане нет проанализированных партий. Вернитесь к скану или запустите новую проверку.",
        action: el("a", { class: "btn btn-secondary btn-sm", href: "#/" }, "Новая проверка"),
      }),
    );
  }

  return card(
    {
      title: "Партии",
      subtitle:
        "По убыванию даты. Строка открывает разбор; ссылка Lichess — публичная партия. Maia не подключена, поэтому обычно «—».",
      actions: badge(`${total}`, "neutral"),
    },
    gamesTable(items, scanId, total, page),
    pagination({
      page,
      pageSize,
      total,
      onNavigate: (target) => loadPage(target),
    }),
  );
}

export async function renderResult(root, params = {}) {
  const scanId = params.scanId;
  if (typeof scanId !== "string" || scanId === "") {
    root.append(
      alert("В адресе нет идентификатора скана. Вернитесь на главную и запустите проверку.", "error", {
        title: "Неверная ссылка",
      }),
    );
    return;
  }

  const header = pageHeader(scanId);
  const verdictSlot = el("div", { class: "verdict-slot" });
  const gamesSlot = el("div", { class: "games-slot" });
  const backHref = `#/scan/${encodeURIComponent(scanId)}`;

  root.append(
    breadcrumbs(scanId),
    header.node,
    verdictSlot,
    gamesSlot,
    el(
      "div",
      { class: "card-footer", style: { borderTop: "0", paddingTop: "0", marginTop: "0" } },
      el("a", { class: "btn btn-secondary", href: backHref }, "К прогрессу скана"),
      el(
        "div",
        { class: "head-actions" },
        el("a", { class: "btn btn-primary", href: "#/" }, "Новая проверка"),
      ),
    ),
  );

  verdictSlot.append(verdictSkeleton());
  gamesSlot.append(gamesSkeleton());

  function redirectToScan() {
    window.setTimeout(() => {
      if (!verdictSlot.isConnected) return;
      window.location.hash = `#/scan/${encodeURIComponent(scanId)}`;
    }, 1400);
  }

  function isMissingScan(error) {
    const detail = typeof error?.detail === "string" ? error.detail : "";
    const message = typeof error?.message === "string" ? error.message : "";
    return /scan not found/i.test(detail) || /scan not found/i.test(message);
  }

  async function loadVerdict() {
    verdictSlot.setAttribute("aria-busy", "true");
    verdictSlot.replaceChildren(verdictSkeleton());
    try {
      const verdict = await getVerdict(scanId);
      if (!verdictSlot.isConnected) return;
      header.eyebrow.textContent = verdict?.model_version
        ? `player-level · ${verdict.model_version}`
        : "player-level";
      verdictSlot.replaceChildren(...verdictCard(verdict));
    } catch (error) {
      if (!verdictSlot.isConnected) return;
      if (error instanceof ApiError && error.status === 409) {
        verdictSlot.replaceChildren(pendingCard(scanId));
        redirectToScan();
        return;
      }
      if (isMissingScan(error)) {
        verdictSlot.replaceChildren(missingScanCard());
        return;
      }
      verdictSlot.replaceChildren(errorCard("Вердикт недоступен", error, loadVerdict));
    } finally {
      if (verdictSlot.isConnected) verdictSlot.setAttribute("aria-busy", "false");
    }
  }

  async function loadGames(page) {
    gamesSlot.setAttribute("aria-busy", "true");
    gamesSlot.replaceChildren(gamesSkeleton());
    try {
      const data = await getGames(scanId, { page, pageSize: PAGE_SIZE });
      if (!gamesSlot.isConnected) return;
      gamesSlot.replaceChildren(gamesCard(data, scanId, page, loadGames));
    } catch (error) {
      if (!gamesSlot.isConnected) return;
      if (isMissingScan(error)) {
        gamesSlot.replaceChildren();
        return;
      }
      gamesSlot.replaceChildren(
        errorCard("Не удалось загрузить партии", error, () => loadGames(page)),
      );
    } finally {
      if (gamesSlot.isConnected) gamesSlot.setAttribute("aria-busy", "false");
    }
  }

  await Promise.allSettled([loadVerdict(), loadGames(1)]);
}
