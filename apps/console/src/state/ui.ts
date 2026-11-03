import { useSyncExternalStore } from "react";

/** Global UI state shared across pages: the settings dialog. */
let settingsOpen = false;
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function openSettings() {
  settingsOpen = true;
  emit();
}

export function setSettingsOpen(open: boolean) {
  settingsOpen = open;
  emit();
}

export function useSettingsOpen() {
  return useSyncExternalStore(subscribe, () => settingsOpen, () => settingsOpen);
}

/** Whether the user asked the OS for reduced motion. */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    (listener) => {
      const query = window.matchMedia("(prefers-reduced-motion: reduce)");
      query.addEventListener("change", listener);
      return () => query.removeEventListener("change", listener);
    },
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    () => false,
  );
}
