import { useState } from "react";
import { useUpdateRouter } from "@/api/queries";
import { isGatewayError } from "@/api/client";
import { INDEX_MODES, POLICIES, type IndexMode, type Policy, type RouterSnapshot, type RouterUpdate } from "@/api/types";
import { Segmented } from "@/components/ui/Segmented";
import { Switch } from "@/components/ui/Switch";
import { useT } from "@/i18n";

/** Policy, index mode and PD switch, all applied with PUT /admin/router. */
export function RouterControls({ snapshot }: { snapshot: RouterSnapshot | undefined }) {
  const t = useT();
  const k = t.cluster.controls;
  const update = useUpdateRouter();
  const [pending, setPending] = useState<RouterUpdate | null>(null);

  const apply = (change: RouterUpdate) => {
    setPending(change);
    update.mutate(change, { onSettled: () => setPending(null) });
  };

  const disabled = !snapshot || update.isPending;
  const error = update.error ? (isGatewayError(update.error) && update.error.unauthorized ? k.token401 : update.error.message) : null;

  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs text-ink-3">{k.policy}</span>
        <Segmented<Policy>
          label={k.policyLabel}
          mono
          value={snapshot?.policy ?? null}
          pending={pending?.policy ?? null}
          disabled={disabled}
          options={POLICIES.map((policy) => ({ value: policy, label: policy, title: t.policyHelp[policy] }))}
          onChange={(policy) => apply({ policy })}
        />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-ink-3">{k.index}</span>
        <Segmented<IndexMode>
          label={k.indexLabel}
          mono
          size="sm"
          value={snapshot?.index_mode ?? null}
          pending={pending?.index_mode ?? null}
          disabled={disabled}
          options={INDEX_MODES.map((mode) => ({ value: mode, label: mode, title: t.indexHelp[mode] }))}
          onChange={(index_mode) => apply({ index_mode })}
        />
      </div>
      <Switch
        label={k.pd}
        stateText={snapshot ? (snapshot.disaggregation ? t.common.on : t.common.off) : undefined}
        checked={snapshot?.disaggregation ?? false}
        disabled={disabled}
        onChange={(disaggregation) => apply({ disaggregation })}
      />
      {error ? (
        <span role="alert" className="text-xs text-bad">
          {k.failed(error)}
        </span>
      ) : null}
    </div>
  );
}
