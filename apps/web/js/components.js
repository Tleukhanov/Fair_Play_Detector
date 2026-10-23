// Small DOM builders for the fair play shell.
//
// Everything renders through createElement/textContent; raw HTML strings are
// never injected. Views receive these nodes from a normal function call, so
// untrusted API values (usernames, summaries) are always treated as text.

/**
 * Create an element.
 *
 * Supported attributes: `class`/`className`, `dataset`, `style` (object),
 * `text`, `on*` event handlers, booleans (rendered as empty attributes) and
 * all regular HTML attributes.
 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class" || key === "className") {
      node.className = String(value);
    } else if (key === "dataset") {
      Object.assign(node.dataset, value);
    } else if (key === "style") {
      if (typeof value === "string") node.setAttribute("style", value);
      else Object.assign(node.style, value);
    } else if (key === "text") {
      node.textContent = String(value);
    } else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) {
      node.setAttribute(key, "");
    } else {
      node.setAttribute(key, String(value));
    }
  }
  // Defence in depth: any link opened in a new tab gets a safe rel, even if a
  // future caller forgets to pass one.
  if (node.getAttribute("target") === "_blank") {
    const rel = new Set((node.getAttribute("rel") || "").split(/\s+/).filter(Boolean));
    rel.add("noopener");
    rel.add("noreferrer");
    node.setAttribute("rel", [...rel].join(" "));
  }
  append(node, children);
  return node;
}

/** DocumentFragment helper: groups children without an extra wrapper. */
export function frag(...children) {
  const node = document.createDocumentFragment();
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false || child === "") continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

function clamp(value, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) return min;
  return Math.min(max, Math.max(min, number));
}

/** Card with an optional header. Accepts a title string or an options object. */
export function card(options = {}, ...children) {
  const opts = typeof options === "string" ? { title: options } : options || {};
  const node = el("section", {
    class: opts.className ? `card ${opts.className}` : "card",
    id: opts.id,
  });
  if (opts.title || opts.subtitle || opts.actions) {
    node.append(
      el(
        "div",
        { class: "card-header" },
        el(
          "div",
          {},
          opts.title ? el("h2", { class: "card-title" }, opts.title) : null,
          opts.subtitle ? el("p", { class: "card-subtitle" }, opts.subtitle) : null,
        ),
        opts.actions ? el("div", { class: "card-actions" }, opts.actions) : null,
      ),
    );
  }
  append(node, children);
  return node;
}

export function badge(text, kind = "neutral") {
  return el("span", { class: `badge badge-${kind}` }, text);
}

export function alert(message, kind = "info", options = {}) {
  const node = el("div", {
    class: `alert alert-${kind}`,
    role: kind === "error" ? "alert" : "status",
  });
  if (options.title) node.append(el("strong", { class: "alert-title" }, options.title));
  node.append(el("div", { class: "alert-body" }, message));
  return node;
}

export function empty(options = {}) {
  const node = el(
    "div",
    { class: "empty" },
    options.icon === null
      ? null
      : el("div", { class: "empty-icon", "aria-hidden": "true" }, options.icon || "♞"),
    options.title ? el("p", { class: "empty-title" }, options.title) : null,
    options.text ? el("p", { class: "empty-text" }, options.text) : null,
  );
  if (options.action) node.append(el("div", { class: "empty-action" }, options.action));
  return node;
}

/** Progress bar. `value` is a 0-100 percentage. */
export function progress(value, options = {}) {
  const percent = clamp(value, 0, 100);
  const wrap = el("div", { class: "progress-block" });
  if (options.label || options.detail) {
    wrap.append(
      el(
        "div",
        { class: "progress-head" },
        el("span", { class: "progress-label" }, options.label || "Прогресс"),
        options.detail ? el("span", { class: "progress-detail" }, options.detail) : null,
      ),
    );
  }
  wrap.append(
    el(
      "div",
      {
        class: "progress",
        role: "progressbar",
        "aria-label": options.label || "Прогресс",
        "aria-valuemin": "0",
        "aria-valuemax": "100",
        "aria-valuenow": String(Math.round(percent)),
        "aria-valuetext": options.detail || `${Math.round(percent)}%`,
      },
      el("div", { class: "progress-bar", style: { width: `${percent}%` } }),
    ),
  );
  if (options.metaLeft || options.metaRight) {
    wrap.append(
      el(
        "div",
        { class: "progress-meta" },
        el("span", {}, options.metaLeft || ""),
        el("span", {}, options.metaRight || ""),
      ),
    );
  }
  return wrap;
}

/** Horizontal stepper. `steps` is a list of strings or {label, detail}. */
export function stepper(steps = [], activeIndex = 0) {
  const list = el("ol", { class: "stepper", "aria-label": "Этапы сканирования" });
  steps.forEach((step, index) => {
    const label = typeof step === "string" ? step : step.label;
    const detail = typeof step === "string" ? null : step.detail;
    const state = index < activeIndex ? "done" : index === activeIndex ? "active" : "todo";
    list.append(
      el(
        "li",
        { class: `step ${state}`, "aria-current": index === activeIndex ? "step" : null },
        el("span", { class: "step-dot", "aria-hidden": "true" }, index < activeIndex ? "✓" : String(index + 1)),
        el(
          "span",
          { class: "step-text" },
          el("span", { class: "step-label" }, label),
          detail ? el("span", { class: "step-detail" }, detail) : null,
        ),
      ),
    );
  });
  return list;
}

/** Skeleton table placeholder. Purely decorative, hidden from screen readers. */
export function skeletonRows(rows = 4, columns = 4) {
  const wrap = el("div", { class: "skeleton-table", "aria-hidden": "true" });
  const widths = ["w-80", "w-60", "w-45", "w-30"];
  for (let row = 0; row < rows; row += 1) {
    const cells = [];
    for (let column = 0; column < columns; column += 1) {
      cells.push(el("span", { class: `skeleton ${widths[(row + column) % widths.length]}` }));
    }
    wrap.append(el("div", { class: "skeleton-row" }, cells));
  }
  return wrap;
}

const RISK_META = {
  low_risk: { label: "низкий риск", kind: "low" },
  suspicious: { label: "подозрительно", kind: "suspicious" },
  high_risk: { label: "высокий риск", kind: "high" },
};

export function riskBadge(level) {
  const meta = RISK_META[level] || { label: String(level || "неизвестно"), kind: "neutral" };
  return el("span", { class: `badge badge-${meta.kind}` }, meta.label);
}

/**
 * Player-level risk meter with threshold zones.
 * Defaults are the TrainedScorer player-mean thresholds documented in README.
 */
export function riskMeter(score, options = {}) {
  const value = clamp(score, 0, 1);
  const lowThreshold = clamp(options.lowThreshold ?? 0.4915, 0, 1);
  const highThreshold = clamp(options.highThreshold ?? 0.5638, lowThreshold, 1);
  const percent = value * 100;
  const lowWidth = lowThreshold * 100;
  const midWidth = Math.max(0, (highThreshold - lowThreshold) * 100);
  const marker = el("span", {
    class: "risk-marker",
    style: { left: `${percent}%` },
    "aria-hidden": "true",
  });
  return el(
    "div",
    { class: "risk-meter-block" },
    el(
      "div",
      { class: "risk-meter-wrap" },
      el(
        "div",
        {
          class: "risk-meter",
          role: "img",
          "aria-label": `Риск ${value.toFixed(2)} из 1.00: низкий до ${lowThreshold.toFixed(2)}, высокий от ${highThreshold.toFixed(2)}`,
        },
        el("span", { class: "risk-zone risk-zone-low", style: { width: `${lowWidth}%` } }),
        el("span", { class: "risk-zone risk-zone-suspicious", style: { width: `${midWidth}%` } }),
        el("span", { class: "risk-zone risk-zone-high" }),
      ),
      marker,
    ),
    el(
      "ul",
      { class: "risk-legend" },
      el("li", {}, el("span", { class: "risk-dot risk-dot-low" }), `низкий < ${lowThreshold.toFixed(2)}`),
      el(
        "li",
        {},
        el("span", { class: "risk-dot risk-dot-suspicious" }),
        `подозрительно ${lowThreshold.toFixed(2)}–${highThreshold.toFixed(2)}`,
      ),
      el("li", {}, el("span", { class: "risk-dot risk-dot-high" }), `высокий ≥ ${highThreshold.toFixed(2)}`),
    ),
  );
}

function formatNumber(value, unit) {
  if (!Number.isFinite(value)) return "—";
  if (unit === "cp") return `${Math.round(value)} cp`;
  if (unit === "s") return `${value.toFixed(1)} с`;
  if (unit === "%") return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(2);
}

/**
 * One evidence row: observed value against the cohort baseline.
 * `item` = {code, label, value, baseline, weight, unit?}.
 */
export function evidenceRow(item) {
  const value = Number(item.value) || 0;
  const baseline = Number(item.baseline) || 0;
  const weight = Number(item.weight);
  const scale = Number(item.scale) > 0 ? Number(item.scale) : Math.max(value, baseline, 1) * 1.2;
  const fillPercent = clamp((value / scale) * 100, 2, 100);
  const baselinePercent = clamp((baseline / scale) * 100, 0, 100);
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
          item.label,
          item.code ? el("span", { class: "evidence-code mono" }, item.code) : null,
        ),
        Number.isFinite(weight) ? el("span", { class: "evidence-weight" }, `вес ${weight.toFixed(2)}`) : null,
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
      el("span", { class: "evidence-value" }, formatNumber(value, item.unit)),
      el("span", { class: "evidence-base" }, `база ${formatNumber(baseline, item.unit)}`),
    ),
  );
}

/**
 * Scrollable table. `headers` are strings or {label, className}; rows are
 * arrays of cells (strings or DOM nodes).
 */
export function tableWrap(headers = [], rows = [], options = {}) {
  const head = el("thead", {}, el("tr", {}, headers.map((header) => {
    const cell = typeof header === "string" ? { label: header } : header;
    return el(
      "th",
      { scope: "col", class: cell.className || null },
      cell.label,
    );
  })));
  const body = el(
    "tbody",
    {},
    rows.map((row) => {
      const cells = row && !Array.isArray(row) ? row.cells : row;
      const rowClass = row && !Array.isArray(row) ? row.className : null;
      return el(
        "tr",
        { class: rowClass || null },
        (cells || []).map((cell, index) => {
          const header = typeof headers[index] === "string" ? {} : headers[index] || {};
          return el("td", { class: header.className || null }, cell);
        }),
      );
    }),
  );
  const table = el("table", { class: options.className || null }, head, body);
  if (options.caption) table.prepend(el("caption", {}, options.caption));
  // Scrollable tables are keyboard-scrollable focus targets on narrow screens.
  return el(
    "div",
    {
      class: "table-wrap",
      tabindex: "0",
      role: "region",
      "aria-label": options.ariaLabel || options.caption || "Таблица",
    },
    table,
  );
}

/**
 * Pagination control. Renders disabled buttons when no handler is supplied,
 * which keeps static previews honest instead of pretending to work.
 */
export function pagination(options = {}) {
  const page = Math.max(1, Number(options.page) || 1);
  const pageSize = Math.max(1, Number(options.pageSize) || 20);
  const total = Math.max(0, Number(options.total) || 0);
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const onNavigate = typeof options.onNavigate === "function" ? options.onNavigate : null;
  const nav = el("nav", { class: "pagination", "aria-label": options.label || "Страницы" });

  if (options.showInfo !== false) {
    nav.append(
      el(
        "span",
        { class: "pagination-info" },
        total > 0 ? `стр. ${page} из ${totalPages} · всего ${total}` : "нет записей",
      ),
    );
  }

  const makeButton = (label, target, extra = {}) => {
    const disabled = !onNavigate || target < 1 || target > totalPages || target === page;
    return el(
      "button",
      {
        type: "button",
        class: `page-btn${target === page ? " active" : ""}`,
        disabled: disabled || null,
        "aria-current": target === page ? "page" : null,
        "aria-label": extra.ariaLabel || `Страница ${target}`,
        onclick: () => {
          if (onNavigate && !disabled) onNavigate(target);
        },
      },
      label,
    );
  };

  nav.append(makeButton("‹", page - 1, { ariaLabel: "Предыдущая страница" }));
  const windowStart = Math.max(1, Math.min(page - 2, totalPages - 4));
  const windowEnd = Math.min(totalPages, windowStart + 4);
  for (let number = windowStart; number <= windowEnd; number += 1) {
    nav.append(makeButton(String(number), number));
  }
  nav.append(makeButton("›", page + 1, { ariaLabel: "Следующая страница" }));
  return nav;
}
