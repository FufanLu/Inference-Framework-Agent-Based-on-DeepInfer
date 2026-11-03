import type { SseFrame } from "./client";
import type { ProtocolName } from "./types";

/**
 * Request builders and stream accumulators for the three protocols the
 * gateway speaks. Field names follow deepseek-recipe's request schemas:
 *   chat_completions  thinking {type} + reasoning_effort, response_format json_schema
 *   responses         instructions + input, reasoning {effort}, text.format json_schema
 *   messages          system, thinking {type} + output_config {effort}; no schema output
 */

export const MODEL_ALIAS = "deepseek-flash";

export type Effort = "low" | "high" | "xhigh" | "max";
export const EFFORTS: readonly Effort[] = ["low", "high", "xhigh", "max"];

export const PROTOCOL_PATH: Record<ProtocolName, string> = {
  chat_completions: "/v1/chat/completions",
  responses: "/v1/responses",
  messages: "/v1/messages",
};

export const PROTOCOL_LABEL: Record<ProtocolName, string> = {
  chat_completions: "Chat Completions",
  responses: "Responses",
  messages: "Anthropic Messages",
};

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface Compose {
  system: string;
  user: string;
  thinking: boolean;
  effort: Effort;
  tools: ToolSpec[];
  /** JSON Schema text for structured output; null when off. */
  schema: Record<string, unknown> | null;
  maxTokens: number | null;
  seed: number | null;
  stream: boolean;
}

/** A chat-completions conversation turn, used by Agent Studio. */
export type ChatMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: string }
  | {
      role: "assistant";
      content: string | null;
      reasoning_content?: string;
      tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
    }
  | { role: "tool"; tool_call_id: string; content: string };

export function buildBody(protocol: ProtocolName, compose: Compose): Record<string, unknown> {
  const { system, user, thinking, effort, tools, schema, maxTokens, seed, stream } = compose;
  switch (protocol) {
    case "chat_completions": {
      const messages: ChatMessage[] = [];
      if (system.trim()) messages.push({ role: "system", content: system });
      messages.push({ role: "user", content: user });
      return chatBody(messages, { thinking, effort, tools, schema, maxTokens, seed, stream });
    }
    case "responses": {
      const body: Record<string, unknown> = {
        model: MODEL_ALIAS,
        input: [{ role: "user", content: user }],
        stream,
        reasoning: { effort: thinking ? effort : "none" },
      };
      if (system.trim()) body.instructions = system;
      if (tools.length > 0) {
        body.tools = tools.map((tool) => ({
          type: "function",
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        }));
      }
      if (schema) body.text = { format: { type: "json_schema", name: "answer", schema } };
      if (maxTokens) body.max_output_tokens = maxTokens;
      return body;
    }
    case "messages": {
      const body: Record<string, unknown> = {
        model: MODEL_ALIAS,
        messages: [{ role: "user", content: user }],
        stream,
        thinking: { type: thinking ? "enabled" : "disabled" },
      };
      if (thinking) body.output_config = { effort };
      if (system.trim()) body.system = system;
      if (tools.length > 0) {
        body.tools = tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          input_schema: tool.parameters,
        }));
      }
      if (maxTokens) body.max_tokens = maxTokens;
      return body;
    }
  }
}

export function chatBody(
  messages: ChatMessage[],
  options: Omit<Compose, "system" | "user">,
): Record<string, unknown> {
  const { thinking, effort, tools, schema, maxTokens, seed, stream } = options;
  const body: Record<string, unknown> = {
    model: MODEL_ALIAS,
    messages,
    stream,
    thinking: { type: thinking ? "enabled" : "disabled" },
  };
  if (stream) body.stream_options = { include_usage: true };
  if (thinking) body.reasoning_effort = effort;
  if (tools.length > 0) {
    body.tools = tools.map((tool) => ({
      type: "function",
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
  }
  if (schema) body.response_format = { type: "json_schema", json_schema: { name: "answer", schema } };
  if (maxTokens) body.max_tokens = maxTokens;
  if (seed !== null) body.seed = seed;
  return body;
}

// ----------------------------------------------------------- accumulation

export interface ToolCallOut {
  id: string;
  name: string;
  arguments: string;
}

export interface Usage {
  prompt: number;
  cached: number;
  completion: number;
  reasoning: number | null;
}

export interface ParsedOutput {
  responseId: string | null;
  reasoning: string;
  content: string;
  toolCalls: ToolCallOut[];
  finishReason: string | null;
  usage: Usage | null;
  /** Error reported inside the stream (after it started). */
  error: string | null;
  /** First reasoning/content/tool delta, ms since the request was sent. */
  firstTokenMs: number | null;
  frames: number;
  /** Responses item id -> tool call index. */
  itemIndex: Record<string, number>;
  /** Messages content block index -> tool call index. */
  blockIndex: Record<number, number>;
}

export function emptyOutput(): ParsedOutput {
  return {
    responseId: null,
    reasoning: "",
    content: "",
    toolCalls: [],
    finishReason: null,
    usage: null,
    error: null,
    firstTokenMs: null,
    frames: 0,
    itemIndex: {},
    blockIndex: {},
  };
}

type Json = Record<string, unknown>;

function asObject(value: unknown): Json | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Json) : null;
}

function str(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function num(value: unknown): number {
  return typeof value === "number" ? value : 0;
}

/** Fold one SSE frame into the parsed output. Returns a new object. */
export function accumulate(protocol: ProtocolName, out: ParsedOutput, frame: SseFrame): ParsedOutput {
  const next: ParsedOutput = {
    ...out,
    frames: out.frames + 1,
    toolCalls: out.toolCalls.slice(),
    itemIndex: { ...out.itemIndex },
    blockIndex: { ...out.blockIndex },
  };
  if (frame.data === "[DONE]") return next;
  let data: Json;
  try {
    data = JSON.parse(frame.data) as Json;
  } catch {
    return next;
  }
  const markToken = () => {
    if (next.firstTokenMs === null) next.firstTokenMs = frame.t;
  };

  if (protocol === "chat_completions") {
    const error = asObject(data.error);
    if (error) {
      next.error = str(error.message) ?? "error in stream";
      return next;
    }
    next.responseId = next.responseId ?? str(data.id);
    const usage = asObject(data.usage);
    if (usage) {
      const details = asObject(usage.completion_tokens_details);
      next.usage = {
        prompt: num(usage.prompt_tokens),
        cached: num(usage.prompt_cache_hit_tokens),
        completion: num(usage.completion_tokens),
        reasoning: details ? num(details.reasoning_tokens) : null,
      };
    }
    const choices = Array.isArray(data.choices) ? data.choices : [];
    for (const raw of choices) {
      const choice = asObject(raw);
      if (!choice) continue;
      const delta = asObject(choice.delta);
      if (delta) {
        const reasoning = str(delta.reasoning_content);
        if (reasoning) {
          next.reasoning += reasoning;
          markToken();
        }
        const content = str(delta.content);
        if (content) {
          next.content += content;
          markToken();
        }
        const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
        for (const rawCall of calls) {
          const call = asObject(rawCall);
          if (!call) continue;
          markToken();
          const index = num(call.index);
          const fn = asObject(call.function);
          const existing = next.toolCalls[index];
          if (!existing) {
            next.toolCalls[index] = {
              id: str(call.id) ?? `call_${index}`,
              name: str(fn?.name) ?? "",
              arguments: str(fn?.arguments) ?? "",
            };
          } else {
            next.toolCalls[index] = { ...existing, arguments: existing.arguments + (str(fn?.arguments) ?? "") };
          }
        }
      }
      const finish = str(choice.finish_reason);
      if (finish) next.finishReason = finish;
    }
    return next;
  }

  if (protocol === "responses") {
    const type = str(data.type) ?? frame.event ?? "";
    switch (type) {
      case "response.created":
      case "response.in_progress": {
        const response = asObject(data.response);
        next.responseId = next.responseId ?? str(response?.id);
        break;
      }
      case "response.reasoning_text.delta":
        next.reasoning += str(data.delta) ?? "";
        markToken();
        break;
      case "response.output_text.delta":
        next.content += str(data.delta) ?? "";
        markToken();
        break;
      case "response.output_item.added": {
        const item = asObject(data.item);
        if (item && item.type === "function_call") {
          markToken();
          next.toolCalls.push({
            id: str(item.call_id) ?? str(item.id) ?? "call",
            name: str(item.name) ?? "",
            arguments: str(item.arguments) ?? "",
          });
          next.itemIndex[str(item.id) ?? ""] = next.toolCalls.length - 1;
        }
        break;
      }
      case "response.function_call_arguments.delta": {
        const index = next.itemIndex[str(data.item_id) ?? ""] ?? next.toolCalls.length - 1;
        const call = next.toolCalls[index];
        if (call) next.toolCalls[index] = { ...call, arguments: call.arguments + (str(data.delta) ?? "") };
        break;
      }
      case "response.function_call_arguments.done": {
        const index = next.itemIndex[str(data.item_id) ?? ""] ?? next.toolCalls.length - 1;
        const call = next.toolCalls[index];
        if (call) next.toolCalls[index] = { ...call, arguments: str(data.arguments) ?? call.arguments };
        break;
      }
      case "response.completed":
      case "response.incomplete": {
        const response = asObject(data.response);
        const usage = asObject(response?.usage);
        if (usage) {
          const input = asObject(usage.input_tokens_details);
          const output = asObject(usage.output_tokens_details);
          next.usage = {
            prompt: num(usage.input_tokens),
            cached: num(input?.cached_tokens),
            completion: num(usage.output_tokens),
            reasoning: output ? num(output.reasoning_tokens) : null,
          };
        }
        const incomplete = asObject(response?.incomplete_details);
        next.finishReason =
          type === "response.incomplete"
            ? `incomplete${incomplete?.reason ? `: ${String(incomplete.reason)}` : ""}`
            : next.toolCalls.length > 0
              ? "function_call"
              : "completed";
        break;
      }
      case "error":
        next.error = str(data.message) ?? "error in stream";
        break;
      default:
        break;
    }
    return next;
  }

  // Anthropic Messages
  const type = str(data.type) ?? frame.event ?? "";
  switch (type) {
    case "message_start": {
      const message = asObject(data.message);
      next.responseId = str(message?.id);
      const usage = asObject(message?.usage);
      if (usage) {
        next.usage = {
          prompt: num(usage.input_tokens) + num(usage.cache_read_input_tokens),
          cached: num(usage.cache_read_input_tokens),
          completion: num(usage.output_tokens),
          reasoning: null,
        };
      }
      break;
    }
    case "content_block_start": {
      const block = asObject(data.content_block);
      if (block?.type === "tool_use") {
        markToken();
        next.blockIndex[num(data.index)] = next.toolCalls.length;
        next.toolCalls.push({ id: str(block.id) ?? "toolu", name: str(block.name) ?? "", arguments: "" });
      }
      break;
    }
    case "content_block_delta": {
      const delta = asObject(data.delta);
      if (!delta) break;
      if (delta.type === "thinking_delta") {
        next.reasoning += str(delta.thinking) ?? "";
        markToken();
      } else if (delta.type === "text_delta") {
        next.content += str(delta.text) ?? "";
        markToken();
      } else if (delta.type === "input_json_delta") {
        const index = next.blockIndex[num(data.index)] ?? next.toolCalls.length - 1;
        const call = next.toolCalls[index];
        if (call) next.toolCalls[index] = { ...call, arguments: call.arguments + (str(delta.partial_json) ?? "") };
      }
      break;
    }
    case "message_delta": {
      const delta = asObject(data.delta);
      const stop = str(delta?.stop_reason);
      if (stop) next.finishReason = stop;
      const usage = asObject(data.usage);
      if (usage) {
        next.usage = {
          prompt: num(usage.input_tokens) + num(usage.cache_read_input_tokens),
          cached: num(usage.cache_read_input_tokens),
          completion: num(usage.output_tokens),
          reasoning: null,
        };
      }
      break;
    }
    case "error": {
      const error = asObject(data.error);
      next.error = str(error?.message) ?? "error in stream";
      break;
    }
    default:
      break;
  }
  return next;
}
