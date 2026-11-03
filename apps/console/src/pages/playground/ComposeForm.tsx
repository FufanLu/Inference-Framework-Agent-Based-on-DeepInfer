import { useId } from "react";
import { EFFORTS, PROTOCOL_LABEL, type Effort } from "@/api/protocols";
import { PROTOCOLS, type ProtocolName } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Field, Input, Textarea } from "@/components/ui/Field";
import { Segmented } from "@/components/ui/Segmented";
import { Switch } from "@/components/ui/Switch";
import { useT } from "@/i18n";
import { WEATHER_TOOL } from "@/lib/demoTools";
import { saveSettings, useSettings } from "@/state/settings";
import { defaultDraft, validate, type Draft } from "./compose";

interface ComposeFormProps {
  draft: Draft;
  onChange: (draft: Draft) => void;
  protocol: ProtocolName;
  onProtocol: (protocol: ProtocolName) => void;
  onSend: () => void;
  onStop: () => void;
  streaming: boolean;
}

export function ComposeForm({ draft, onChange, protocol, onProtocol, onSend, onStop, streaming }: ComposeFormProps) {
  const t = useT();
  const k = t.playground.form;
  const settings = useSettings();
  const ids = { system: useId(), user: useId(), schema: useId(), max: useId(), seed: useId(), key: useId() };
  const { compose, errors } = validate(draft, t);
  const set = <K extends keyof Draft>(key: K, value: Draft[K]) => onChange({ ...draft, [key]: value });
  const schemaDropped = protocol === "messages" && draft.schemaOn;

  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(event) => {
        event.preventDefault();
        if (compose && !streaming) onSend();
      }}
    >
      <Field label={k.protocol}>
        <Segmented<ProtocolName>
          label={k.protocol}
          value={protocol}
          onChange={onProtocol}
          options={PROTOCOLS.map((p) => ({ value: p, label: PROTOCOL_LABEL[p] }))}
          size="sm"
          className="self-start"
        />
      </Field>

      <Field label={k.system} htmlFor={ids.system}>
        <Textarea id={ids.system} rows={2} value={draft.system} onChange={(event) => set("system", event.target.value)} />
      </Field>
      <Field label={k.user} htmlFor={ids.user} error={errors.user}>
        <Textarea
          id={ids.user}
          rows={3}
          value={draft.user}
          onChange={(event) => set("user", event.target.value)}
          aria-invalid={Boolean(errors.user)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              event.preventDefault();
              if (compose && !streaming) onSend();
            }
          }}
        />
      </Field>

      <div className="flex flex-col gap-3 border-t border-rule pt-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Switch label={k.thinking} checked={draft.thinking} onChange={(value) => set("thinking", value)} stateText={draft.thinking ? t.common.on : t.common.off} />
          <Segmented<Effort>
            label={k.effort}
            size="sm"
            mono
            value={draft.effort}
            disabled={!draft.thinking}
            onChange={(value) => set("effort", value)}
            options={EFFORTS.map((effort) => ({ value: effort, label: effort }))}
          />
        </div>
        <Switch label={k.tool(WEATHER_TOOL.name)} checked={draft.tool} onChange={(value) => set("tool", value)} stateText={draft.tool ? k.toolOn : k.toolOff} />
        <Switch label={k.schema} checked={draft.schemaOn} onChange={(value) => set("schemaOn", value)} stateText={draft.schemaOn ? t.common.on : t.common.off} />
        {draft.schemaOn ? (
          <Field label={k.schemaField} htmlFor={ids.schema} error={errors.schema}>
            <Textarea
              id={ids.schema}
              rows={7}
              value={draft.schemaText}
              onChange={(event) => set("schemaText", event.target.value)}
              className="font-mono text-xs"
              spellCheck={false}
              aria-invalid={Boolean(errors.schema)}
            />
          </Field>
        ) : null}
      </div>

      <div className="grid grid-cols-2 gap-3 border-t border-rule pt-3">
        <Field label={k.maxTokens} htmlFor={ids.max} error={errors.maxTokens} hint={k.maxHint}>
          <Input
            id={ids.max}
            inputMode="numeric"
            value={draft.maxTokens}
            onChange={(event) => set("maxTokens", event.target.value)}
            placeholder={k.auto}
            className="font-mono"
            aria-invalid={Boolean(errors.maxTokens)}
          />
        </Field>
        <Field label={k.seed} htmlFor={ids.seed} error={errors.seed} hint={k.seedHint}>
          <Input
            id={ids.seed}
            inputMode="numeric"
            value={draft.seed}
            onChange={(event) => set("seed", event.target.value)}
            placeholder={k.seedPlaceholder}
            className="font-mono"
            aria-invalid={Boolean(errors.seed)}
          />
        </Field>
        <Field label={k.apiKey} htmlFor={ids.key} hint={k.apiKeyHint} className="col-span-2">
          <Input
            id={ids.key}
            type="password"
            value={settings.apiKey}
            onChange={(event) => saveSettings({ ...settings, apiKey: event.target.value.trim() })}
            placeholder={k.apiKeyPlaceholder}
            autoComplete="off"
            className="font-mono"
          />
        </Field>
      </div>

      {schemaDropped ? <p className="text-xs text-warn">{k.messagesNoSchema}</p> : null}

      <div className="flex items-center gap-2 border-t-[1.5px] border-ink pt-3">
        {streaming ? (
          <Button variant="danger" onClick={onStop}>
            {k.stop}
          </Button>
        ) : (
          <Button type="submit" variant="primary" disabled={!compose}>
            {k.send}
          </Button>
        )}
        <Button variant="ghost" onClick={() => onChange(defaultDraft(t))} disabled={streaming}>
          {k.reset}
        </Button>
        <span className="ml-auto hidden text-2xs text-ink-3 xl:inline">{k.shortcut}</span>
      </div>
    </form>
  );
}
