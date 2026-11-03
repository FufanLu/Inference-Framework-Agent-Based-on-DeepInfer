import { useEffect, useMemo, useRef, useState } from "react";
import { isGatewayError, streamInference, type SseFrame } from "@/api/client";
import { accumulate, buildBody, emptyOutput, PROTOCOL_PATH } from "@/api/protocols";
import type { ProtocolName } from "@/api/types";
import { PageHeader } from "@/components/PageHeader";
import { TabsContent, TabsList, TabsRoot, TabsTrigger } from "@/components/ui/Tabs";
import { getT, useT } from "@/i18n";
import { ComposeForm } from "./playground/ComposeForm";
import { defaultDraft, validate, type Draft } from "./playground/compose";
import { FrameLog } from "./playground/FrameLog";
import { Inspector } from "./playground/Inspector";
import { OutputView, type Run } from "./playground/OutputView";
import { ProtocolCompare } from "./playground/ProtocolCompare";

const FRAME_CAP = 2_000;

export function PlaygroundPage() {
  const t = useT();
  const [draft, setDraft] = useState<Draft>(() => defaultDraft(getT()));
  const [protocol, setProtocol] = useState<ProtocolName>("chat_completions");
  const [tab, setTab] = useState("output");
  const [run, setRun] = useState<Run | null>(null);
  const [frames, setFrames] = useState<SseFrame[]>([]);
  const [truncated, setTruncated] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const { compose } = useMemo(() => validate(draft, t), [draft, t]);

  useEffect(() => () => controller.current?.abort(), []);

  async function send() {
    if (!compose) return;
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    const body = buildBody(protocol, compose);
    const started = performance.now();
    let output = emptyOutput();
    let pendingFrames: SseFrame[] = [];
    let allFrames: SseFrame[] = [];
    let dropped = 0;
    let scheduled = false;
    const flush = () => {
      scheduled = false;
      allFrames = allFrames.concat(pendingFrames);
      pendingFrames = [];
      if (allFrames.length > FRAME_CAP) {
        dropped += allFrames.length - FRAME_CAP;
        allFrames = allFrames.slice(-FRAME_CAP);
      }
      setFrames(allFrames);
      setTruncated(dropped);
      setRun((previous) => (previous ? { ...previous, output } : previous));
    };

    setFrames([]);
    setTruncated(0);
    setRun({
      protocol,
      body,
      startedAt: Date.now(),
      endedAt: null,
      status: "streaming",
      output,
      httpStatus: null,
      requestId: null,
      headersMs: null,
      totalMs: null,
      error: null,
      errorCode: null,
      retryAfter: null,
    });

    try {
      const result = await streamInference(
        PROTOCOL_PATH[protocol],
        body,
        (frame) => {
          output = accumulate(protocol, output, frame);
          pendingFrames.push(frame);
          if (!scheduled) {
            scheduled = true;
            requestAnimationFrame(flush);
          }
        },
        abort.signal,
      );
      flush();
      setRun((previous) =>
        previous
          ? {
              ...previous,
              output,
              status: result.error || output.error ? "error" : "done",
              endedAt: Date.now(),
              httpStatus: result.status,
              requestId: result.requestId,
              headersMs: result.headersMs,
              totalMs: performance.now() - started,
              error: result.error?.message ?? null,
              errorCode: result.error?.code ?? null,
              retryAfter: result.error?.retryAfter ?? null,
            }
          : previous,
      );
    } catch (error) {
      flush();
      const aborted = isGatewayError(error) && error.kind === "aborted";
      setRun((previous) =>
        previous
          ? {
              ...previous,
              output,
              status: aborted ? "aborted" : "error",
              endedAt: Date.now(),
              totalMs: performance.now() - started,
              error: aborted ? null : error instanceof Error ? error.message : String(error),
            }
          : previous,
      );
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title={t.playground.title} description={t.playground.description} />
      <div className="grid items-start gap-x-8 gap-y-8 lg:grid-cols-[minmax(300px,360px)_minmax(0,1fr)]">
        <div className="lg:sticky lg:top-5">
          <ComposeForm
            draft={draft}
            onChange={setDraft}
            protocol={protocol}
            onProtocol={setProtocol}
            onSend={() => {
              setTab((current) => (current === "frames" ? current : "output"));
              void send();
            }}
            onStop={() => controller.current?.abort()}
            streaming={run?.status === "streaming"}
          />
        </div>
        <TabsRoot value={tab} onValueChange={setTab} className="min-w-0">
          <TabsList label={t.playground.tabs.label} className="mb-5">
            <TabsTrigger value="output">{t.playground.tabs.output}</TabsTrigger>
            <TabsTrigger value="frames" count={run ? frames.length + truncated : undefined}>
              {t.playground.tabs.frames}
            </TabsTrigger>
            <TabsTrigger value="inspector">{t.playground.tabs.inspector}</TabsTrigger>
            <TabsTrigger value="compare">{t.playground.tabs.compare}</TabsTrigger>
          </TabsList>
          <TabsContent value="output">
            <OutputView run={run} expectJson={Boolean(compose?.schema)} />
          </TabsContent>
          <TabsContent value="frames">
            <FrameLog frames={frames} truncated={truncated} />
          </TabsContent>
          <TabsContent value="inspector">
            <Inspector compose={compose} protocol={protocol} />
          </TabsContent>
          <TabsContent value="compare">
            <ProtocolCompare compose={compose} />
          </TabsContent>
        </TabsRoot>
      </div>
    </div>
  );
}
