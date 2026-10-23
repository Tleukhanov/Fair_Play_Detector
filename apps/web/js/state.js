// Local browser state for the fair play detector.
//
// There is no server-side history and no account: the last scans live in
// localStorage only. Every storage access is guarded so private mode or a
// disabled storage backend cannot break the views.

const RECENT_KEY = "fpd.recentScans.v1";
const MAX_RECENT = 5;

function getStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function normalizeEntry(entry) {
  if (!entry || typeof entry !== "object") return null;
  const scanId = typeof entry.scanId === "string" ? entry.scanId.trim() : "";
  const username = typeof entry.username === "string" ? entry.username.trim() : "";
  if (!scanId || !username) return null;

  const parsed = typeof entry.createdAt === "string" ? Date.parse(entry.createdAt) : Number.NaN;
  const createdAt = Number.isNaN(parsed) ? new Date().toISOString() : new Date(parsed).toISOString();
  const status =
    typeof entry.status === "string" && entry.status.trim() ? entry.status.trim() : "queued";

  return { scanId, username, createdAt, status };
}

function readEntries() {
  const storage = getStorage();
  if (!storage) return [];
  try {
    const raw = storage.getItem(RECENT_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map(normalizeEntry)
      .filter(Boolean)
      .slice(0, MAX_RECENT);
  } catch {
    return [];
  }
}

function writeEntries(entries) {
  const trimmed = entries.slice(0, MAX_RECENT);
  const storage = getStorage();
  if (!storage) return trimmed;
  try {
    storage.setItem(RECENT_KEY, JSON.stringify(trimmed));
  } catch {
    // Quota or private mode: the in-memory list is still returned to the view.
  }
  return trimmed;
}

/** Last scans, newest first (maximum 5). */
export function loadRecent() {
  return readEntries();
}

/**
 * Insert or replace a scan at the top of the recent list.
 * Accepts `{scanId, username, createdAt, status}`.
 */
export function saveRecent(entry) {
  const item = normalizeEntry(entry);
  if (!item) return readEntries();
  const next = readEntries().filter((existing) => existing.scanId !== item.scanId);
  next.unshift(item);
  return writeEntries(next);
}

/** Update the stored status of one scan; a no-op when the scan is unknown. */
export function updateRecentStatus(scanId, status) {
  if (typeof scanId !== "string" || !scanId) return readEntries();
  const value = typeof status === "string" && status.trim() ? status.trim() : "queued";
  let changed = false;
  const next = readEntries().map((entry) => {
    if (entry.scanId !== scanId) return entry;
    changed = true;
    return { ...entry, status: value };
  });
  return changed ? writeEntries(next) : readEntries();
}

/** Remove every stored scan. */
export function clearRecent() {
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.removeItem(RECENT_KEY);
  } catch {
    // Ignore: nothing to clear when storage is unavailable.
  }
}
