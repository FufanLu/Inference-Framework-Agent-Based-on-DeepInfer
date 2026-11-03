import type { Compose, Effort } from "@/api/protocols";
import type { Messages } from "@/i18n";
import { WEATHER_TOOL } from "@/lib/demoTools";

/** What the form holds; converted to a `Compose` before sending. */
export interface Draft {
  system: string;
  user: string;
  thinking: boolean;
  effort: Effort;
  tool: boolean;
  schemaOn: boolean;
  schemaText: string;
  maxTokens: string;
  seed: string;
}

export const DEFAULT_SCHEMA = JSON.stringify(
  {
    type: "object",
    properties: {
      city: { type: "string" },
      condition: { type: "string", enum: ["sunny", "cloudy", "rain", "snow"] },
      temperature_c: { type: "number" },
    },
    required: ["city", "condition", "temperature_c"],
  },
  null,
  2,
);

/** The starting example; in Chinese it is docs/illustrated/02's 18-token prompt. */
export const defaultDraft = (t: Messages): Draft => ({
  system: t.playground.defaults.system,
  user: t.playground.defaults.user,
  thinking: false,
  effort: "high",
  tool: false,
  schemaOn: false,
  schemaText: DEFAULT_SCHEMA,
  maxTokens: "",
  seed: "",
});

export interface DraftErrors {
  schema?: string;
  maxTokens?: string;
  seed?: string;
  user?: string;
}

export function validate(draft: Draft, t: Messages): { compose: Compose | null; errors: DraftErrors } {
  const k = t.playground.form;
  const errors: DraftErrors = {};
  let schema: Record<string, unknown> | null = null;
  if (draft.schemaOn) {
    try {
      const parsed: unknown = JSON.parse(draft.schemaText);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) errors.schema = k.schemaNotObject;
      else schema = parsed as Record<string, unknown>;
    } catch (error) {
      errors.schema = k.schemaInvalid(error instanceof Error ? error.message : String(error));
    }
  }
  let maxTokens: number | null = null;
  if (draft.maxTokens.trim()) {
    const value = Number(draft.maxTokens);
    if (!Number.isInteger(value) || value <= 0) errors.maxTokens = k.positiveInt;
    else maxTokens = value;
  }
  let seed: number | null = null;
  if (draft.seed.trim()) {
    const value = Number(draft.seed);
    if (!Number.isSafeInteger(value) || value < 0) errors.seed = k.seedInvalid;
    else seed = value;
  }
  if (!draft.user.trim()) errors.user = k.userEmpty;
  const ok = Object.keys(errors).length === 0;
  return {
    errors,
    compose: ok
      ? {
          system: draft.system,
          user: draft.user,
          thinking: draft.thinking,
          effort: draft.effort,
          tools: draft.tool ? [WEATHER_TOOL] : [],
          schema,
          maxTokens,
          seed,
          stream: true,
        }
      : null,
  };
}
