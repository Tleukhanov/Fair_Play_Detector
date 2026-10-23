// Scan view: live progress while polling GET /api/v1/scans/{id}.
//
// The view owns no server state besides the poll itself. It stops polling on
// terminal statuses, on navigation away, or when the rendered card leaves the
// document, and redirects to the verdict route on completion.

import { pollScan } from "../api.js";
import { alert, badge, card, el, progress, stepper } from "../components.js";
import { loadRecent, updateRecentStatus } from "../state.js";

const STAGE_ORDER = ["queued", "fetching", "analyzing", "scoring", "explaining", "completed"];

const STAGE_LABELS = {
  queued: "В очереди",
  fetching: "Скачиваем партии",
  analyzing: "Анализируем Stockfish",
  scoring: "Считаем риск",
  explaining: "Формируем вердикт",
  completed: "Готово",
  failed: "Ошибка",
};

const STAGE_KIND = {
  queued: "neutral",
  fetching: "blue",
  analyzing: "blue",
  scoring: "blue",
  explaining: "blue",
  completed: "good",
  failed: "bad",
};

const ERROR_MESSAGES = {
  INGEST_FAILED:
    "Не удалось получить партии из Lichess. Проверьте ник/контроль или попробуйте позже.",
  ANALYSIS_FAILED: "Stockfish не смог проанализировать партии.",
  SCORING_FAILED: "Недостаточно пригодных партий: нужно минимум 10.",
  STALE_SCAN: "Скан устарел. Запустите проверку заново.",
  QUEUE_FULL: "Очередь переполнена. Попробуйте позже.",
};

const SCAN_ID_RE = /^[0-9A-Za-z-]{1,64}$/;

function shortId(scanId) {
  return scanId.length > 12 ? `${scanId.slice(0, 8)}…` : scanId;
}

function formatDuration(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function statusKey(status) {
  return typeof status?.status === "string" && status.status ? status.status : "queued";
}

function errorMessage(status) {
  const code = typeof status?.error_code === "string" ? status.error_code.trim() : "";
  if (code && ERROR_MESSAGES[code]) return ERROR_MESSAGES[code];
  const message = typeof status?.error_message === "string" ? status.error_message.trim() : "";
  return message || "Не удалось завершить проверку. Запустите её заново.";
}

function hashPath() {
  const path = (window.location.hash.replace(/^#/, "").split("?")[0] || "/").replace(/\/+$/, "");
  return path || "/";
}

function renderMissingScan(root, message) {
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
      { class: "page-head" },
      el("p", { class: "eyebrow" }, "сканирование"),
      el("h1", {}, "Скан не найден"),
      el("p", { class: "lead" }, message),
    ),
    card(
      { title: "Нечего показывать" },
      alert("Откройте форму на главной странице и запустите новую проверку.", "error"),
      el(
        "div",
        { class: "form-actions" },
        el("a", { class: "btn btn-primary", href: "#/" }, "На главную"),
      ),
    ),
  );
}

export async function renderScan(root, params = {}) {
  const scanId = typeof params.scanId === "string" ? params.scanId : "";

  if (!SCAN_ID_RE.test(scanId)) {
    renderMissingScan(root, "В адресе нет корректного идентификатора скана.");
    return;
  }

  const stored = loadRecent().find((entry) => entry.scanId === scanId);
  const username = stored?.username || "";
  const startedAt = Date.now();

  const statusBadge = badge(STAGE_LABELS.queued, STAGE_KIND.queued);
  const heading = el("h1", {}, "Проверка выполняется");
  const lead = el(
    "p",
    { class: "lead" },
    username
      ? `Загружаем публичные партии ${username} и анализируем их Stockfish. Страница обновляется автоматически.`
      : "Загружаем публичные партии Lichess и анализируем их Stockfish. Страница обновляется автоматически.",
  );

  const progressHost = el("div", { "aria-live": "polite" });
  const stepperHost = el("div");
  const errorHost = el("div");
  const elapsedNode = el("span", { class: "mono" }, "0:00");

  const progressCard = card(
    {
      title: "Прогресс",
      subtitle: username ? `Игрок: ${username}` : `Скан ${shortId(scanId)}`,
    },
    progressHost,
  );
  progressCard.setAttribute("aria-busy", "true");

  const stepperCard = card(
    { title: "Этапы", subtitle: "Пайплайн скана: от загрузки PGN до вердикта." },
    stepperHost,
  );

  let stopPolling = null;
  let timerId = null;
  let cleanedUp = false;
  let lastStageIndex = 0;

  function cleanup() {
    if (cleanedUp) return;
    cleanedUp = true;
    if (stopPolling) {
      stopPolling();
      stopPolling = null;
    }
    if (timerId !== null) {
      clearInterval(timerId);
      timerId = null;
    }
    window.removeEventListener("hashchange", handleHashChange);
    progressCard.setAttribute("aria-busy", "false");
  }

  function handleHashChange() {
    if (hashPath() !== `/scan/${scanId}`) cleanup();
  }

  function setBadge(key, label) {
    statusBadge.textContent = label || STAGE_LABELS[key] || "Проверка";
    statusBadge.className = `badge badge-${STAGE_KIND[key] || "neutral"}`;
  }

  function renderProgress(status) {
    const key = statusKey(status);
    const found = Math.max(0, Math.floor(Number(status?.games_found) || 0));
    const analyzed = Math.max(0, Math.floor(Number(status?.games_analyzed) || 0));

    let percent = 8;
    if (key === "completed") percent = 100;
    else if (found > 0) percent = Math.min(100, (analyzed / found) * 100);

    let label = STAGE_LABELS[key] || "Проверка";
    if (found === 0 && (key === "queued" || key === "fetching")) {
      label = "Скачиваем партии…";
    }

    const detail =
      found > 0
        ? `${analyzed} из ${found} партий`
        : key === "queued"
          ? "ожидаем свободный воркер"
          : "партии ещё не загружены";

    progressHost.replaceChildren(
      progress(percent, {
        label,
        detail,
        metaLeft: `найдено ${found} / проанализировано ${analyzed}`,
        metaRight: el("span", {}, "прошло ", elapsedNode),
      }),
    );
  }

  function renderStepper(key, status) {
    const index = STAGE_ORDER.indexOf(key);
    if (index >= 0 && key !== "completed") lastStageIndex = Math.max(lastStageIndex, index);

    if (key === "failed") {
      const passed = STAGE_ORDER.slice(
        0,
        Math.max(1, Math.min(lastStageIndex + 1, STAGE_ORDER.length - 1)),
      ).map((stage) => ({ label: STAGE_LABELS[stage] }));
      const code = typeof status?.error_code === "string" ? status.error_code : "";
      passed.push({ label: STAGE_LABELS.failed, detail: code || undefined });
      stepperHost.replaceChildren(stepper(passed, passed.length - 1));
      return;
    }

    const steps = STAGE_ORDER.map((stage) => ({ label: STAGE_LABELS[stage] }));
    const activeIndex = key === "completed" ? steps.length : Math.max(0, index);
    stepperHost.replaceChildren(stepper(steps, activeIndex));
  }

  function renderFailed(status) {
    const code = typeof status?.error_code === "string" ? status.error_code.trim() : "";
    const homeLink = el("a", { class: "btn btn-primary", href: "#/" }, "На главную");
    const retryLink = username
      ? el(
          "a",
          {
            class: "btn btn-secondary",
            href: `#/?username=${encodeURIComponent(username)}`,
            "aria-label": `Запустить новую проверку для ${username}`,
          },
          "Повторить",
        )
      : null;

    errorHost.replaceChildren(
      alert(errorMessage(status), "error", { title: "Не удалось завершить проверку" }),
      code
        ? el(
            "details",
            { class: "small muted" },
            el("summary", {}, "Код ошибки"),
            el("p", { class: "mono" }, code),
          )
        : null,
      el("div", { class: "form-actions", style: { marginTop: "12px" } }, homeLink, retryLink),
    );
  }

  function rerender() {
    cleanup();
    root.replaceChildren();
    Promise.resolve(renderScan(root, { scanId })).catch((error) => {
      console.error("scan view re-render failed", error);
      root.append(
        alert("Не удалось перезапустить экран скана. Обновите страницу.", "error", {
          title: "Ошибка интерфейса",
        }),
      );
    });
  }

  function onUpdate(status) {
    if (!progressCard.isConnected) {
      cleanup();
      return;
    }
    const key = statusKey(status);

    if (key === "completed") {
      setBadge("completed");
      heading.textContent = "Проверка завершена";
      renderProgress(status);
      renderStepper("completed", status);
      updateRecentStatus(scanId, "completed");
      cleanup();
      window.location.hash = `#/scan/${scanId}/result`;
      return;
    }

    if (key === "failed") {
      setBadge("failed");
      heading.textContent = "Проверка не удалась";
      lead.textContent = "Скан остановлен. Ниже — причина и варианты действий.";
      renderProgress(status);
      renderStepper("failed", status);
      updateRecentStatus(scanId, "failed");
      renderFailed(status);
      cleanup();
      return;
    }

    setBadge(key);
    renderProgress(status);
    renderStepper(key, status);
  }

  function onError() {
    if (!progressCard.isConnected) {
      cleanup();
      return;
    }
    cleanup();
    statusBadge.textContent = "нет связи";
    statusBadge.className = "badge badge-bad";
    heading.textContent = "Нет связи с сервером";
    errorHost.replaceChildren(
      alert(
        "Не удалось получить состояние скана: сервер не отвечает. Проверьте подключение и повторите.",
        "error",
        { title: "Сетевая ошибка" },
      ),
      el(
        "div",
        { class: "form-actions" },
        el("button", { type: "button", class: "btn btn-primary", onclick: rerender }, "Повторить"),
        el("a", { class: "btn btn-ghost", href: "#/" }, "На главную"),
      ),
    );
  }

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
        el("p", { class: "eyebrow" }, username ? `скан ${shortId(scanId)} · ${username}` : `скан ${shortId(scanId)}`),
        heading,
        lead,
      ),
      el(
        "div",
        { class: "head-actions" },
        statusBadge,
        el("a", { class: "btn btn-secondary btn-sm", href: "#/" }, "Новая проверка"),
      ),
    ),
    progressCard,
    errorHost,
    stepperCard,
  );

  renderProgress({ status: "queued", games_found: 0, games_analyzed: 0 });
  renderStepper("queued", { status: "queued" });

  timerId = setInterval(() => {
    if (!progressCard.isConnected) {
      cleanup();
      return;
    }
    elapsedNode.textContent = formatDuration(Date.now() - startedAt);
  }, 1000);

  if (hashPath() !== `/scan/${scanId}`) {
    // The view was replaced before polling started (extremely unlikely, but
    // avoids a stray poller after the initial render).
    cleanup();
    return;
  }

  window.addEventListener("hashchange", handleHashChange);
  stopPolling = pollScan(scanId, { onUpdate, onError });
}
