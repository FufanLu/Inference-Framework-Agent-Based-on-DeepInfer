import { isGatewayError, streamInference } from "@/api/client";
import { accumulate, chatBody, emptyOutput, PROTOCOL_PATH, type ChatMessage, type Effort, type ParsedOutput } from "@/api/protocols";
import { getT } from "@/i18n";
import { findTool, type DemoTool } from "@/lib/demoTools";

/**
 * A browser-side agent loop over /v1/chat/completions: call the model with
 * tools, run the tools it asks for here in the page, send the results back,
 * repeat until it answers in text.
 */

export interface ModelStep {
  kind: "model";
  turn: number;
  /** ms since the run started. */
  start: number;
  end: number | null;
  firstToken: number | null;
  requestId: string | null;
  output: ParsedOutput;
  httpStatus: number | null;
  error: string | null;
  errorCode: string | null;
  retryAfter: number | null;
  /** Wall clock when the call ended, for Retry-After countdowns. */
  endedAt: number | null;
}

export interface ToolStep {
  kind: "tool";
  turn: number;
  start: number;
  end: number | null;
  callId: string;
  name: string;
  rawArguments: string;
  result: unknown;
  error: string | null;
}

export type AgentStep = ModelStep | ToolStep;

export interface AgentRun {
  status: "running" | "done" | "error" | "aborted" | "exhausted";
  steps: AgentStep[];
  answer: string | null;
  /** ms since start, when finished. */
  total: number | null;
  error: string | null;
}

export interface AgentConfig {
  task: string;
  system: string;
  tools: DemoTool[];
  thinking: boolean;
  effort: Effort;
  maxTurns: number;
}

export async function runAgent(config: AgentConfig, onUpdate: (run: AgentRun) => void, signal: AbortSignal): Promise<AgentRun> {
  const t0 = performance.now();
  const now = () => performance.now() - t0;
  const run: AgentRun = { status: "running", steps: [], answer: null, total: null, error: null };
  const publish = () => onUpdate({ ...run, steps: run.steps.map((step) => ({ ...step })) });
  let scheduled = false;
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      publish();
    });
  };

  const messages: ChatMessage[] = [];
  if (config.system.trim()) messages.push({ role: "system", content: config.system });
  messages.push({ role: "user", content: config.task });

  try {
    for (let turn = 1; turn <= config.maxTurns; turn += 1) {
      const step: ModelStep = {
        kind: "model",
        turn,
        start: now(),
        end: null,
        firstToken: null,
        requestId: null,
        output: emptyOutput(),
        httpStatus: null,
        error: null,
        errorCode: null,
        retryAfter: null,
        endedAt: null,
      };
      run.steps.push(step);
      publish();
      const body = chatBody(messages, {
        thinking: config.thinking,
        effort: config.effort,
        tools: config.tools.map((tool) => tool.spec),
        schema: null,
        maxTokens: null,
        seed: null,
        stream: true,
      });
      const result = await streamInference(
        PROTOCOL_PATH.chat_completions,
        body,
        (frame) => {
          step.output = accumulate("chat_completions", step.output, frame);
          if (step.firstToken === null && step.output.firstTokenMs !== null) step.firstToken = step.start + step.output.firstTokenMs;
          schedule();
        },
        signal,
      );
      step.end = now();
      step.endedAt = Date.now();
      step.requestId = result.requestId;
      step.httpStatus = result.status;
      if (result.error || step.output.error) {
        step.error = result.error?.message ?? step.output.error;
        step.errorCode = result.error?.code ?? null;
        step.retryAfter = result.error?.retryAfter ?? null;
        run.status = "error";
        run.error = step.error;
        break;
      }
      const calls = step.output.toolCalls;
      messages.push({
        role: "assistant",
        content: step.output.content || null,
        ...(step.output.reasoning ? { reasoning_content: step.output.reasoning } : {}),
        ...(calls.length > 0
          ? { tool_calls: calls.map((call) => ({ id: call.id, type: "function" as const, function: { name: call.name, arguments: call.arguments } })) }
          : {}),
      });
      if (calls.length === 0) {
        run.status = "done";
        run.answer = step.output.content;
        break;
      }
      for (const call of calls) {
        const tool: ToolStep = {
          kind: "tool",
          turn,
          start: now(),
          end: null,
          callId: call.id,
          name: call.name,
          rawArguments: call.arguments,
          result: null,
          error: null,
        };
        const impl = findTool(call.name);
        try {
          const args = call.arguments.trim() ? (JSON.parse(call.arguments) as Record<string, unknown>) : {};
          tool.result = impl ? impl.run(args) : { error: `tool ${call.name} does not exist` };
          if (!impl) tool.error = getT().agent.toolMissing(call.name);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          tool.error = getT().agent.badArgs(message);
          tool.result = { error: `arguments are not valid JSON: ${message}` };
        }
        tool.end = now();
        run.steps.push(tool);
        messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(tool.result) });
        publish();
      }
      if (turn === config.maxTurns) run.status = "exhausted";
    }
  } catch (error) {
    const last = run.steps[run.steps.length - 1];
    if (last && last.end === null) last.end = now();
    if (isGatewayError(error) && error.kind === "aborted") run.status = "aborted";
    else {
      run.status = "error";
      run.error = error instanceof Error ? error.message : String(error);
    }
  }
  run.total = now();
  publish();
  return run;
}
