// Home view: scan form, recent scans and the probabilistic disclaimer.
//
// Submit flow: validate locally -> POST /api/v1/scans -> remember the scan in
// localStorage -> route to the polling view. Errors are reported without
// leaving the page, and the submit button is always restored.

import { createScan } from "../api.js";
import { alert, badge, card, el, empty } from "../components.js";
import { clearRecent, loadRecent, saveRecent } from "../state.js";

const USERNAME_RE = /^[A-Za-z0-9_][A-Za-z0-9_-]*$/;
const USERNAME_MAX = 30;
const MAX_GAMES = [10, 20, 30, 40, 50];
const SPEEDS = ["blitz", "rapid"];
const SUBMIT_LABEL = "Запустить проверку";

const STATUS_META = {
  queued: { label: "в очереди", kind: "neutral" },
  fetching: { label: "скачиваем", kind: "blue" },
  analyzing: { label: "анализ", kind: "blue" },
  scoring: { label: "скоринг", kind: "blue" },
  explaining: { label: "вердикт", kind: "blue" },
  completed: { label: "готово", kind: "good" },
  failed: { label: "ошибка", kind: "bad" },
};

const DATE_FORMATTER = (() => {
  try {
    return new Intl.DateTimeFormat("ru-RU", {
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return null;
  }
})();

function statusMeta(status) {
  return STATUS_META[status] || { label: status || "неизвестно", kind: "neutral" };
}

function formatDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  if (DATE_FORMATTER) {
    try {
      return DATE_FORMATTER.format(date);
    } catch {
      // Fall through to a plain locale string.
    }
  }
  try {
    return date.toLocaleString("ru-RU");
  } catch {
    return iso;
  }
}

/** Prefill from `#/?username=...`, written by the retry button on the scan view. */
function prefillUsername() {
  try {
    const query = window.location.hash.split("?")[1] || "";
    const value = new URLSearchParams(query).get("username") || "";
    return value.trim().slice(0, USERNAME_MAX);
  } catch {
    return "";
  }
}

function speedOption(value, label, checked) {
  return el(
    "label",
    { class: "seg-item" },
    el("input", { type: "radio", name: "speed", value, checked: checked || null }),
    el("span", {}, label),
  );
}

function setFieldError(input, messageNode, message) {
  if (messageNode) {
    messageNode.textContent = message || "";
    messageNode.hidden = !message;
  }
  if (input) {
    if (message) input.setAttribute("aria-invalid", "true");
    else input.removeAttribute("aria-invalid");
  }
}

function submitErrorMessage(error) {
  const status = Number(error?.status) || 0;
  const message = typeof error?.message === "string" ? error.message.trim() : "";

  if (status === 422) {
    return "Сервер отклонил параметры запроса. Проверьте ник, число партий и контроль времени.";
  }
  if (status === 429) return "Слишком много запросов. Подождите минуту и попробуйте снова.";
  if (status === 503) return "Очередь переполнена. Попробуйте через минуту.";
  if (status === 0) return "Сервер недоступен. Проверьте подключение и попробуйте ещё раз.";
  if (status >= 500) return `Ошибка сервера (${status}). Попробуйте ещё раз позже.`;
  return message || "Не удалось запустить проверку. Попробуйте ещё раз.";
}

function buildForm() {
  const formAlert = el("div", { class: "form-alert" });

  const usernameInput = el("input", {
    id: "username",
    name: "username",
    type: "text",
    required: true,
    maxlength: String(USERNAME_MAX),
    placeholder: "lichess nick",
    autocomplete: "off",
    autocapitalize: "none",
    autocorrect: "off",
    spellcheck: "false",
    "aria-describedby": "username-help username-error",
    value: prefillUsername(),
  });
  const usernameError = el("p", {
    class: "form-help",
    id: "username-error",
    hidden: true,
  });

  const usernameGroup = el(
    "div",
    { class: "form-group" },
    el("label", { for: "username" }, "Ник на Lichess"),
    usernameInput,
    el(
      "p",
      { class: "form-help", id: "username-help" },
      "Публичный профиль Lichess. Регистрация и доступ к аккаунту не нужны.",
    ),
    usernameError,
  );

  const speedInputs = [speedOption("blitz", "Блиц", true), speedOption("rapid", "Рапид", false)];
  const speedError = el("p", { class: "form-help", id: "speed-error", hidden: true });
  const speedGroup = el(
    "fieldset",
    { class: "form-group", "aria-describedby": "speed-error" },
    el("legend", {}, "Контроль времени"),
    el("div", { class: "segmented" }, speedInputs),
    speedError,
  );

  const maxGamesSelect = el(
    "select",
    { id: "max-games", name: "max_games", "aria-describedby": "max-games-error" },
    MAX_GAMES.map((count) =>
      el("option", { value: String(count), selected: count === 30 || null }, String(count)),
    ),
  );
  const maxGamesError = el("p", {
    class: "form-help",
    id: "max-games-error",
    hidden: true,
  });
  const maxGamesGroup = el(
    "div",
    { class: "form-group form-group-narrow" },
    el("label", { for: "max-games" }, "Сколько партий"),
    maxGamesSelect,
    maxGamesError,
  );

  const ratedOnlyInput = el("input", {
    id: "rated-only",
    type: "checkbox",
    name: "rated_only",
    checked: true,
  });
  const ratedGroup = el(
    "div",
    { class: "form-group form-group-check" },
    el(
      "label",
      { class: "checkbox", for: "rated-only" },
      ratedOnlyInput,
      "Только рейтинговые партии",
    ),
  );

  const submitButton = el(
    "button",
    { type: "submit", class: "btn btn-primary" },
    SUBMIT_LABEL,
  );

  const form = el(
    "form",
    {
      id: "scan-form",
      class: "scan-form",
      autocomplete: "off",
      novalidate: true,
    },
    formAlert,
    usernameGroup,
    speedGroup,
    el("div", { class: "form-row" }, maxGamesGroup, ratedGroup),
    el(
      "div",
      { class: "form-actions" },
      submitButton,
      el("a", { class: "btn btn-ghost", href: "#/methodology" }, "Как считается риск"),
    ),
    el("p", { class: "form-note" }, "Обычно занимает 1–2 минуты. Вкладку можно не закрывать."),
  );

  const clearErrors = () => {
    formAlert.replaceChildren();
    setFieldError(usernameInput, usernameError, "");
    setFieldError(maxGamesSelect, maxGamesError, "");
    for (const radio of form.querySelectorAll('input[name="speed"]')) {
      radio.removeAttribute("aria-invalid");
    }
    setFieldError(null, speedError, "");
  };

  usernameInput.addEventListener("input", () => {
    setFieldError(usernameInput, usernameError, "");
  });
  maxGamesSelect.addEventListener("change", () => {
    setFieldError(maxGamesSelect, maxGamesError, "");
  });
  const speedRadios = form.querySelectorAll('input[name="speed"]');
  for (const radio of speedRadios) {
    radio.addEventListener("change", () => {
      for (const item of speedRadios) item.removeAttribute("aria-invalid");
      setFieldError(null, speedError, "");
    });
  }

  let submitting = false;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitting) return;
    clearErrors();

    const username = usernameInput.value.trim();
    const maxGames = Number(maxGamesSelect.value);
    const checkedSpeed = form.querySelector('input[name="speed"]:checked');
    const speed = checkedSpeed ? checkedSpeed.value : "";
    const ratedOnly = ratedOnlyInput.checked;

    const errors = [];
    if (!username) {
      errors.push({ input: usernameInput, node: usernameError, message: "Укажите ник на Lichess." });
    } else if (!USERNAME_RE.test(username)) {
      errors.push({
        input: usernameInput,
        node: usernameError,
        message: "Ник может содержать латинские буквы, цифры, дефис и подчёркивание.",
      });
    }
    if (!Number.isInteger(maxGames) || maxGames < 10 || maxGames > 50) {
      errors.push({
        input: maxGamesSelect,
        node: maxGamesError,
        message: "Выберите от 10 до 50 партий.",
      });
    }
    if (!SPEEDS.includes(speed)) {
      errors.push({
        input: form.querySelector('input[name="speed"]'),
        node: speedError,
        message: "Выберите блиц или рапид.",
      });
    }

    if (errors.length) {
      for (const error of errors) setFieldError(error.input, error.node, error.message);
      formAlert.replaceChildren(
        alert("Проверьте выделенные поля и попробуйте снова.", "error", {
          title: "Форма заполнена неверно",
        }),
      );
      (errors[0].input || form).focus?.();
      return;
    }

    submitting = true;
    submitButton.disabled = true;
    submitButton.textContent = "Запускаем…";
    submitButton.setAttribute("aria-busy", "true");
    form.setAttribute("aria-busy", "true");

    try {
      const response = await createScan({ username, maxGames, speed, ratedOnly });
      const scanId = typeof response?.scan_id === "string" ? response.scan_id : "";
      if (!scanId) throw new Error("Сервер не вернул идентификатор скана.");

      saveRecent({
        scanId,
        username,
        createdAt: new Date().toISOString(),
        status: typeof response?.status === "string" ? response.status : "queued",
      });
      // The router replaces this view; keep the button disabled to avoid a
      // second POST while the hash change is processed.
      window.location.hash = `#/scan/${encodeURIComponent(scanId)}`;
    } catch (error) {
      submitting = false;
      submitButton.disabled = false;
      submitButton.textContent = SUBMIT_LABEL;
      submitButton.removeAttribute("aria-busy");
      form.setAttribute("aria-busy", "false");
      formAlert.replaceChildren(
        alert(submitErrorMessage(error), "error", { title: "Не удалось запустить проверку" }),
      );
    }
  });

  return form;
}

function recentItem(entry) {
  const meta = statusMeta(entry.status);
  return el(
    "li",
    {},
    el(
      "a",
      {
        href: `#/scan/${encodeURIComponent(entry.scanId)}`,
        "aria-label": `Открыть скан ${entry.username}`,
      },
      entry.username,
    ),
    " ",
    badge(meta.label, meta.kind),
    " ",
    el("span", { class: "muted small" }, formatDate(entry.createdAt)),
  );
}

function buildRecentCard(onClear) {
  const entries = loadRecent();
  const actions = entries.length
    ? el("button", { type: "button", class: "btn btn-ghost btn-sm", onclick: onClear }, "Очистить")
    : null;
  const body = entries.length
    ? el("ul", { class: "list" }, entries.map((entry) => recentItem(entry)))
    : empty({
        title: "Пока нет проверок",
        text: "Запустите первую проверку — она появится в этом списке.",
      });

  return card(
    {
      title: "Недавние проверки",
      subtitle: "Хранятся только в этом браузере.",
      actions,
    },
    body,
  );
}

function recentSection() {
  const host = el("div");
  const refresh = () => {
    host.replaceChildren(
      buildRecentCard(() => {
        clearRecent();
        refresh();
      }),
    );
  };
  refresh();
  return host;
}

export async function renderHome(root) {
  root.append(
    el(
      "section",
      { class: "hero" },
      el(
        "div",
        { class: "hero-copy" },
        el("p", { class: "eyebrow" }, "вероятностный анализ · Lichess"),
        el(
          "h1",
          { class: "hero-title" },
          "Проверить, насколько партии похожи на машинные",
        ),
        el(
          "p",
          { class: "lead" },
          "Сигнал по статистике публичных партий: совпадения ходов с Stockfish, потеря оценки и время на ход. Без регистрации, без доступа к аккаунту и без установки.",
        ),
        el(
          "ul",
          { class: "hero-points" },
          el("li", {}, "10–50 последних блиц- или рапид-партий"),
          el("li", {}, "игровой уровень: среднее по партиям + bootstrap-интервал"),
          el("li", {}, "вердикт — вероятность, а не обвинение"),
        ),
        el(
          "div",
          { class: "hero-actions" },
          el("a", { class: "btn btn-ghost", href: "#/methodology" }, "Методика и ограничения"),
        ),
      ),
      card(
        {
          className: "scan-card",
          title: "Новая проверка",
          subtitle: "Данные берутся из публичного API Lichess.",
        },
        buildForm(),
      ),
    ),
    recentSection(),
    alert("Это вероятностный сигнал, не доказательство читерства", "warn", {
      title: "Как читать результат",
    }),
  );
}
