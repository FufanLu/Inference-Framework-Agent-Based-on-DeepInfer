import { useSyncExternalStore } from "react";
import { en, type Messages } from "./en";
import { zh } from "./zh";

export type Locale = "en" | "zh";
export type { Messages };

const LOCALE_KEY = "leviathan.console.locale.v1";
const DICTIONARIES: Record<Locale, Messages> = { en, zh };

function loadLocale(): Locale {
  try {
    return window.localStorage.getItem(LOCALE_KEY) === "zh" ? "zh" : "en";
  } catch {
    return "en";
  }
}

let locale: Locale = loadLocale();
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getLocale(): Locale {
  return locale;
}

/** The dictionary of the current locale, for code outside React. */
export function getT(): Messages {
  return DICTIONARIES[locale];
}

export function setLocale(next: Locale) {
  locale = next;
  try {
    if (next === "en") window.localStorage.removeItem(LOCALE_KEY);
    else window.localStorage.setItem(LOCALE_KEY, next);
  } catch {
    // Storage unavailable: the choice lasts for this page view.
  }
  applyLocale();
  for (const listener of listeners) listener();
}

/** Reflect the locale on <html lang>. */
export function applyLocale() {
  document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
}

export function useLocale(): Locale {
  return useSyncExternalStore(subscribe, getLocale, getLocale);
}

/** The current dictionary; components re-render when the locale changes. */
export function useT(): Messages {
  return DICTIONARIES[useLocale()];
}
