import { useSyncExternalStore } from "react";

/** Connection settings, persisted in localStorage. */
export interface Settings {
  /** Base URL of the gateway, without a trailing slash. */
  gatewayUrl: string;
  /** Bearer token for /admin endpoints (gateway `auth.admin_token`). */
  adminToken: string;
  /** API key for /v1 endpoints (gateway `auth.api_keys`). */
  apiKey: string;
}

export type ThemePreference = "system" | "light" | "dark";

const SETTINGS_KEY = "leviathan.console.settings.v1";
const THEME_KEY = "leviathan.console.theme.v1";

const envUrl = (import.meta.env.VITE_GATEWAY_URL as string | undefined)?.trim();
export const DEFAULT_GATEWAY_URL = normalizeUrl(envUrl || "http://localhost:8080");

export function normalizeUrl(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

function readJson<T>(key: string): Partial<T> | null {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as Partial<T>) : null;
  } catch {
    return null;
  }
}

function writeString(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable (private mode, blocked site data). The
    // settings then live for this page view only.
  }
}

function loadSettings(): Settings {
  const stored = readJson<Settings>(SETTINGS_KEY);
  return {
    gatewayUrl: normalizeUrl(stored?.gatewayUrl || DEFAULT_GATEWAY_URL),
    adminToken: stored?.adminToken ?? "",
    apiKey: stored?.apiKey ?? "",
  };
}

function loadTheme(): ThemePreference {
  try {
    const value = window.localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

type Listener = () => void;

let settings: Settings = loadSettings();
let theme: ThemePreference = loadTheme();
const listeners = new Set<Listener>();

function emit() {
  for (const listener of listeners) listener();
}

function subscribe(listener: Listener) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Subscribe to settings and theme changes outside React. */
export const subscribeSettings = subscribe;

export function getSettings(): Settings {
  return settings;
}

export function saveSettings(next: Settings) {
  settings = { ...next, gatewayUrl: normalizeUrl(next.gatewayUrl || DEFAULT_GATEWAY_URL) };
  const isDefault =
    settings.gatewayUrl === DEFAULT_GATEWAY_URL && !settings.adminToken && !settings.apiKey;
  writeString(SETTINGS_KEY, isDefault ? null : JSON.stringify(settings));
  emit();
}

export function useSettings(): Settings {
  return useSyncExternalStore(subscribe, getSettings, getSettings);
}

export function getTheme(): ThemePreference {
  return theme;
}

export function setTheme(next: ThemePreference) {
  theme = next;
  writeString(THEME_KEY, next === "system" ? null : next);
  applyTheme();
  emit();
}

export function useTheme(): ThemePreference {
  return useSyncExternalStore(subscribe, getTheme, getTheme);
}

/** Reflect the theme preference on <html data-theme>. */
export function applyTheme() {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
}
