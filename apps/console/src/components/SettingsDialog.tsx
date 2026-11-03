import { Dialog } from "radix-ui";
import { X } from "lucide-react";
import { useState } from "react";
import { probeGateway, type ProbeResult } from "@/api/client";
import { setLocale, useLocale, useT, type Locale } from "@/i18n";
import { DEFAULT_GATEWAY_URL, getSettings, saveSettings, setTheme, useTheme, type ThemePreference } from "@/state/settings";
import { setSettingsOpen, useSettingsOpen } from "@/state/ui";
import { Button } from "./ui/Button";
import { Field, Input } from "./ui/Field";
import { Segmented } from "./ui/Segmented";
import { StatusMark } from "./ui/StatusMark";

function SettingsForm({ onDone }: { onDone: () => void }) {
  const t = useT();
  const locale = useLocale();
  const initial = getSettings();
  const theme = useTheme();
  const [gatewayUrl, setGatewayUrl] = useState(initial.gatewayUrl);
  const [adminToken, setAdminToken] = useState(initial.adminToken);
  const [apiKey, setApiKey] = useState(initial.apiKey);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [probing, setProbing] = useState(false);

  const urlValid = /^https?:\/\/[^\s/]+/.test(gatewayUrl.trim());
  const themes = [
    { value: "system", label: t.theme.system },
    { value: "light", label: t.theme.light },
    { value: "dark", label: t.theme.dark },
  ] as const;
  const locales = [
    { value: "en", label: "English" },
    { value: "zh", label: "中文" },
  ] as const;

  async function runProbe() {
    setProbing(true);
    setProbe(await probeGateway(gatewayUrl, adminToken));
    setProbing(false);
  }

  return (
    <form
      className="flex flex-col gap-5"
      onSubmit={(event) => {
        event.preventDefault();
        if (!urlValid) return;
        saveSettings({ gatewayUrl, adminToken: adminToken.trim(), apiKey: apiKey.trim() });
        onDone();
      }}
    >
      <Field
        label={t.settings.gatewayUrl}
        htmlFor="gateway-url"
        error={urlValid ? null : t.settings.urlInvalid}
        hint={
          <>
            {t.settings.buildDefault} <code className="font-mono">{DEFAULT_GATEWAY_URL}</code> (VITE_GATEWAY_URL)
          </>
        }
      >
        <Input
          id="gateway-url"
          value={gatewayUrl}
          onChange={(event) => {
            setGatewayUrl(event.target.value);
            setProbe(null);
          }}
          spellCheck={false}
          autoComplete="url"
          className="font-mono"
          aria-invalid={!urlValid}
        />
      </Field>
      <Field label={t.settings.adminToken} htmlFor="admin-token" hint={t.settings.adminHint}>
        <Input
          id="admin-token"
          type="password"
          value={adminToken}
          onChange={(event) => {
            setAdminToken(event.target.value);
            setProbe(null);
          }}
          autoComplete="off"
          className="font-mono"
          placeholder={t.settings.tokenPlaceholder}
        />
      </Field>
      <Field label={t.settings.apiKey} htmlFor="api-key" hint={t.settings.apiKeyHint}>
        <Input
          id="api-key"
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          autoComplete="off"
          className="font-mono"
          placeholder={t.settings.apiKeyPlaceholder}
        />
      </Field>
      <div className="flex flex-wrap gap-x-8 gap-y-4">
        <Field label={t.theme.label}>
          <Segmented<ThemePreference> label={t.theme.label} value={theme} options={themes} onChange={setTheme} size="sm" />
        </Field>
        <Field label={t.locale.label}>
          <Segmented<Locale> label={t.locale.label} value={locale} options={locales} onChange={setLocale} size="sm" />
        </Field>
      </div>

      <div className="border-t border-rule pt-4">
        <div className="flex flex-wrap items-center gap-3">
          <Button size="sm" onClick={runProbe} busy={probing} disabled={!urlValid}>
            {t.settings.test}
          </Button>
          {probe ? (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
              <StatusMark
                state={probe.reachable ? "live" : "down"}
                tone={probe.reachable ? "ok" : "bad"}
                label={probe.reachable ? t.settings.health(probe.health ?? "") : t.settings.unreachable}
              />
              {probe.reachable ? (
                <StatusMark
                  state={probe.admin === "ok" ? "live" : "down"}
                  tone={probe.admin === "ok" ? "ok" : "bad"}
                  label={
                    probe.admin === "ok"
                      ? t.settings.adminOk(probe.version ?? "?")
                      : probe.admin === "unauthorized"
                        ? t.settings.adminUnauthorized
                        : t.settings.adminFailed(probe.adminDetail ?? "")
                  }
                />
              ) : null}
            </div>
          ) : null}
        </div>
      </div>

      <div className="flex justify-end gap-2">
        <Dialog.Close asChild>
          <Button>{t.common.cancel}</Button>
        </Dialog.Close>
        <Button type="submit" variant="primary" disabled={!urlValid}>
          {t.common.save}
        </Button>
      </div>
    </form>
  );
}

export function SettingsDialog() {
  const t = useT();
  const open = useSettingsOpen();
  return (
    <Dialog.Root open={open} onOpenChange={setSettingsOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="anim-overlay fixed inset-0 z-40 bg-[var(--overlay)]" />
        <Dialog.Content className="anim-dialog fixed top-1/2 left-1/2 z-50 max-h-[92dvh] w-[min(540px,92vw)] -translate-x-1/2 -translate-y-1/2 overflow-y-auto border border-ink bg-sheet shadow-[var(--shadow)] focus:outline-none">
          <header className="flex items-start justify-between border-b-[1.5px] border-ink px-6 pt-5 pb-3">
            <div>
              <Dialog.Title className="cond-tight text-xl font-bold">{t.settings.title}</Dialog.Title>
              <Dialog.Description className="mt-0.5 text-xs text-ink-3">{t.settings.description}</Dialog.Description>
            </div>
            <Dialog.Close
              className="-mr-2 inline-flex size-8 items-center justify-center rounded-sm text-ink-2 hover:bg-wash"
              aria-label={t.common.close}
            >
              <X className="size-4" />
            </Dialog.Close>
          </header>
          <div className="px-6 py-5">{open ? <SettingsForm onDone={() => setSettingsOpen(false)} /> : null}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
