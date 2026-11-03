import { useId, useState } from "react";
import { useUpsertTenant } from "@/api/queries";
import type { Tenant } from "@/api/types";
import { ErrorState } from "@/components/States";
import { Button } from "@/components/ui/Button";
import { Field, Input } from "@/components/ui/Field";
import { Switch } from "@/components/ui/Switch";
import { useT } from "@/i18n";

function parseLimit(text: string): number | null | "invalid" {
  if (!text.trim()) return null;
  const value = Number(text.replace(/[,_\s]/g, ""));
  return Number.isSafeInteger(value) && value > 0 ? value : "invalid";
}

/** Create or edit a tenant: POST /admin/tenants replaces the whole record. */
export function TenantForm({ tenant, onDone }: { tenant: Tenant | null; onDone: () => void }) {
  const t = useT();
  const k = t.tenants.form;
  const ids = { id: useId(), name: useId(), rpm: useId(), tpm: useId(), quota: useId() };
  const save = useUpsertTenant();
  const [id, setId] = useState(tenant?.id ?? "");
  const [name, setName] = useState(tenant?.name ?? "");
  const [rpm, setRpm] = useState(tenant?.rpm ? String(tenant.rpm) : "");
  const [tpm, setTpm] = useState(tenant?.tpm ? String(tenant.tpm) : "");
  const [quota, setQuota] = useState(tenant?.monthly_token_quota ? String(tenant.monthly_token_quota) : "");
  const [disabled, setDisabled] = useState(tenant?.disabled ?? false);

  const idValid = /^[A-Za-z0-9_-]{1,64}$/.test(id) && id !== "anonymous";
  const limits = { rpm: parseLimit(rpm), tpm: parseLimit(tpm), quota: parseLimit(quota) };
  const valid = idValid && name.trim().length > 0 && Object.values(limits).every((value) => value !== "invalid");

  return (
    <form
      className="flex flex-col gap-5"
      onSubmit={(event) => {
        event.preventDefault();
        if (!valid) return;
        save.mutate(
          {
            id,
            name: name.trim(),
            rpm: limits.rpm === "invalid" ? null : limits.rpm,
            tpm: limits.tpm === "invalid" ? null : limits.tpm,
            monthly_token_quota: limits.quota === "invalid" ? null : limits.quota,
            disabled,
          },
          { onSuccess: onDone },
        );
      }}
    >
      <Field label={k.id} htmlFor={ids.id} hint={k.idHint} error={id && !idValid ? k.idInvalid : null}>
        <Input
          id={ids.id}
          value={id}
          onChange={(event) => setId(event.target.value.trim())}
          readOnly={tenant !== null}
          className="font-mono"
          spellCheck={false}
          autoComplete="off"
          aria-invalid={Boolean(id) && !idValid}
        />
      </Field>
      <Field label={k.name} htmlFor={ids.name} error={name || !id ? null : k.nameRequired}>
        <Input id={ids.name} value={name} onChange={(event) => setName(event.target.value)} autoComplete="off" />
      </Field>
      <div className="grid grid-cols-2 gap-4">
        <Field label={k.rpm} htmlFor={ids.rpm} hint={k.limitHint} error={limits.rpm === "invalid" ? k.limitInvalid : null}>
          <Input id={ids.rpm} inputMode="numeric" value={rpm} onChange={(event) => setRpm(event.target.value)} placeholder={t.tenants.unlimited} className="font-mono" />
        </Field>
        <Field label={k.tpm} htmlFor={ids.tpm} hint={k.limitHint} error={limits.tpm === "invalid" ? k.limitInvalid : null}>
          <Input id={ids.tpm} inputMode="numeric" value={tpm} onChange={(event) => setTpm(event.target.value)} placeholder={t.tenants.unlimited} className="font-mono" />
        </Field>
        <Field label={k.quota} htmlFor={ids.quota} hint={k.limitHint} error={limits.quota === "invalid" ? k.limitInvalid : null} className="col-span-2">
          <Input id={ids.quota} inputMode="numeric" value={quota} onChange={(event) => setQuota(event.target.value)} placeholder={t.tenants.unlimited} className="font-mono" />
        </Field>
      </div>
      <div className="flex flex-col gap-1">
        <Switch label={k.disabled} checked={disabled} onChange={setDisabled} stateText={disabled ? t.tenants.disabled : t.tenants.active} />
        <p className="pl-[42px] text-xs text-ink-3">{k.disabledHint}</p>
      </div>
      {save.error ? <ErrorState error={save.error} what={k.failed} compact /> : null}
      <div className="flex justify-end gap-2 border-t-[1.5px] border-ink pt-3">
        <Button onClick={onDone}>{t.common.cancel}</Button>
        <Button type="submit" variant="primary" disabled={!valid} busy={save.isPending}>
          {k.save}
        </Button>
      </div>
    </form>
  );
}
