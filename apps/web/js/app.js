// fair play detector — hash router and theme controller.
//
// Router contract (stable across waves): every view exports
// `async function render(root, params)` and appends its markup to `root`.
// Fetch/scan orchestration stays in the Scan Flow wave.

import { alert } from "./components.js";
import { renderGame } from "./views/game.js";
import { renderHome } from "./views/home.js";
import { renderMethodology } from "./views/methodology.js";
import { renderResult } from "./views/result.js";
import { renderScan } from "./views/scan.js";

const SITE_NAME = "fair play detector";
const THEME_KEY = "theme";

export const API_BASE =
  document.querySelector('meta[name="api-base"]')?.getAttribute("content") || "/api/v1";

const ROUTES = [
  {
    name: "home",
    pattern: /^\/?$/,
    title: "Проверка партий Lichess",
    nav: "home",
    render: renderHome,
  },
  {
    name: "methodology",
    pattern: /^\/methodology\/?$/,
    title: "Методология",
    nav: "methodology",
    render: renderMethodology,
  },
  {
    name: "result",
    pattern: /^\/scan\/([0-9A-Za-z-]+)\/result\/?$/,
    title: "Вердикт",
    nav: "home",
    render: renderResult,
    params: (match) => ({ scanId: match[1] }),
  },
  {
    name: "game",
    pattern: /^\/scan\/([0-9A-Za-z-]+)\/game\/([0-9A-Za-z_-]+)\/?$/,
    title: "Партия",
    nav: "home",
    render: renderGame,
    params: (match) => ({ scanId: match[1], gameId: match[2] }),
  },
  {
    name: "scan",
    pattern: /^\/scan\/([0-9A-Za-z-]+)\/?$/,
    title: "Сканирование",
    nav: "home",
    render: renderScan,
    params: (match) => ({ scanId: match[1] }),
  },
];

// --- Theme ---------------------------------------------------------------

function readStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return stored === "dark" || stored === "light" ? stored : null;
  } catch {
    return null;
  }
}

function systemTheme() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const button = document.querySelector("[data-theme-toggle]");
  if (!button) return;
  const dark = theme === "dark";
  button.textContent = dark ? "☀" : "☾";
  button.title = dark ? "Светлая тема" : "Тёмная тема";
  button.setAttribute("aria-label", dark ? "Светлая тема" : "Тёмная тема");
}

function setupTheme() {
  applyTheme(readStoredTheme() || systemTheme());
  const button = document.querySelector("[data-theme-toggle]");
  if (!button) return;
  button.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // Private mode / storage disabled: the theme still applies for this view.
    }
  });

  const media = window.matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener?.("change", () => {
    if (!readStoredTheme()) applyTheme(systemTheme());
  });
}

// --- Router --------------------------------------------------------------

function currentPath() {
  const hash = window.location.hash.replace(/^#/, "");
  const path = (hash.split("?")[0] || "/").trim();
  return path.startsWith("/") ? path : `/${path}`;
}

function matchRoute(path) {
  for (const route of ROUTES) {
    const match = route.pattern.exec(path);
    if (match) {
      return { route, params: route.params ? route.params(match) : {} };
    }
  }
  return null;
}

function setActiveNav(name) {
  document.querySelectorAll("[data-nav]").forEach((link) => {
    const active = link.getAttribute("data-nav") === name;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

function updateTitle(route) {
  document.title = `${route.title} · ${SITE_NAME}`;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) {
    meta.setAttribute(
      "content",
      document.documentElement.dataset.theme === "dark" ? "#161512" : "#e8e5e2",
    );
  }
}

async function renderRoute() {
  const root = document.getElementById("app");
  if (!root) return;

  const path = currentPath();
  const matched = matchRoute(path);
  if (!matched && path !== "/") {
    history.replaceState(null, "", "#/");
  }
  const route = matched?.route || ROUTES[0];
  const params = matched?.params || {};

  setActiveNav(route.nav);
  updateTitle(route);
  root.classList.remove("view-enter");
  root.setAttribute("aria-busy", "true");
  root.replaceChildren();

  try {
    await route.render(root, params);
  } catch (error) {
    console.error("view render failed", error);
    root.replaceChildren(
      alert(
        "Не удалось отобразить страницу. Обновите её или вернитесь на главную.",
        "error",
        { title: "Ошибка интерфейса" },
      ),
    );
  } finally {
    root.setAttribute("aria-busy", "false");
    // Restart the enter animation after the new content is in the DOM.
    void root.offsetWidth;
    root.classList.add("view-enter");
  }
}

function start() {
  setupTheme();
  if (!window.location.hash) {
    history.replaceState(null, "", "#/");
  }
  window.addEventListener("hashchange", () => {
    renderRoute();
    window.scrollTo({ top: 0, behavior: "auto" });
  });
  renderRoute();
}

start();
