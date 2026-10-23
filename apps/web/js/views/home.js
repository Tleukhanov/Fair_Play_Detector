// Home view: static scan form shell. The Scan Flow wave wires submit -> POST
// /api/v1/scans and the polling redirect; keep the form ids and field names.

import { alert, card, el } from "../components.js";

const MAX_GAMES = [10, 20, 30, 40, 50];

function speedOption(value, label, checked) {
  return el(
    "label",
    { class: "seg-item" },
    el("input", {
      type: "radio",
      name: "speed",
      value,
      checked: checked || null,
    }),
    el("span", {}, label),
  );
}

function buildForm() {
  const form = el("form", {
    id: "scan-form",
    class: "scan-form",
    autocomplete: "off",
    novalidate: true,
  });

  form.append(
    el(
      "div",
      { class: "form-group" },
      el("label", { for: "username" }, "Ник на Lichess"),
      el("input", {
        id: "username",
        name: "username",
        type: "text",
        required: true,
        maxlength: "30",
        placeholder: "например, DrNykterstein",
        autocomplete: "off",
        spellcheck: "false",
        "aria-describedby": "username-help",
      }),
      el(
        "p",
        { class: "form-help", id: "username-help" },
        "Публичный профиль Lichess. Регистрация и доступ к аккаунту не нужны.",
      ),
    ),
    el(
      "fieldset",
      { class: "form-group" },
      el("legend", {}, "Контроль времени"),
      el(
        "div",
        { class: "segmented" },
        speedOption("blitz", "Блиц", true),
        speedOption("rapid", "Рапид", false),
      ),
    ),
    el(
      "div",
      { class: "form-row" },
      el(
        "div",
        { class: "form-group", style: { flex: "0 1 180px" } },
        el("label", { for: "max-games" }, "Сколько партий"),
        el(
          "select",
          { id: "max-games", name: "max_games" },
          MAX_GAMES.map((count) =>
            el("option", { value: String(count), selected: count === 30 || null }, String(count)),
          ),
        ),
      ),
      el(
        "div",
        { class: "form-group", style: { flex: "1 1 240px", alignSelf: "flex-end" } },
        el(
          "label",
          { class: "checkbox", for: "rated-only" },
          el("input", { id: "rated-only", type: "checkbox", name: "rated_only", checked: true }),
          "Только рейтинговые партии",
        ),
      ),
    ),
    el(
      "div",
      { class: "form-actions" },
      el("button", { type: "submit", class: "btn btn-primary" }, "Запустить проверку"),
      el("a", { class: "btn btn-ghost", href: "#/methodology" }, "Как считается риск"),
    ),
    el(
      "p",
      { class: "form-note" },
      "Визуальный каркас: отправка формы подключится в следующем коммите.",
    ),
  );

  let stubNoticeShown = false;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    // Stub for the Scan Flow wave: validate, submit, then route to #/scan/<id>.
    const username = form.querySelector("#username");
    if (username && !username.value.trim()) {
      username.setAttribute("aria-invalid", "true");
      username.reportValidity?.();
      username.focus();
      return;
    }
    username?.removeAttribute("aria-invalid");
    if (!stubNoticeShown) {
      stubNoticeShown = true;
      const notice = alert(
        "Форма пока не отправляет запрос: логика сканирования появится в следующем коммите.",
        "info",
        { title: "Статический предпросмотр" },
      );
      notice.classList.add("form-stub-note");
      form.append(notice);
    }
  });

  return form;
}

function feature(title, text) {
  return el("section", { class: "card feature" }, el("h3", {}, title), el("p", {}, text));
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
          el("a", { class: "btn btn-secondary", href: "#/scan/demo/result" }, "Посмотреть демо-вердикт"),
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
    el(
      "section",
      { class: "grid grid-3" },
      feature(
        "Только публичные данные",
        "Скан читает открытые партии через Lichess API. Приватные данные, cookies и доступ к аккаунту не используются.",
      ),
      feature(
        "Прозрачные признаки",
        "Доля совпадений с top-1/9/15 Stockfish, средний CPL, разброс оценки и время на ход. Пороги и метрики опубликованы.",
      ),
      feature(
        "Сигнал, а не приговор",
        "Результат не доказывает читерство и не предназначен для автомодерации. Ложноположительные срабатывания возможны.",
      ),
    ),
    alert(
      "Сервис показывает статистическую вероятность машиноподобной игры на выбранной выборке. Он не устанавливает факт нарушения правил.",
      "warn",
      { title: "Как читать результат" },
    ),
  );
}
