/**
 * Letta stream messages that the Agent SDK passes through as raw
 * `stream_event`s, parsed into what the adapter reports to ACP clients.
 */

export interface UsageReport {
  /** Tokens in the context window after this step. */
  contextTokens: number;
  /** Cost of this step in USD, when the server reports it. */
  costUsd?: number;
}

export interface CompactionStart {
  kind: "compaction_start";
  trigger?: string;
}

export interface CompactionSummary {
  kind: "compaction_summary";
  summary: string;
  contextTokensBefore?: number;
  contextTokensAfter?: number;
}

export interface RetryReport {
  attempt: number;
  maxAttempts: number;
  delayMs: number;
  reason?: string;
}

export type LettaStreamEvent =
  | { kind: "usage"; usage: UsageReport }
  | CompactionStart
  | CompactionSummary
  | { kind: "retry"; retry: RetryReport };

export function parseStreamEvent(event: unknown): LettaStreamEvent | null {
  if (!event || typeof event !== "object") return null;
  const raw = event as Record<string, unknown>;
  switch (raw.message_type) {
    case "usage_statistics": {
      // prompt_tokens is a fair stand-in only for a single step; a run-level
      // aggregate sums every step's prompt and would overstate the context.
      const contextTokens =
        number(raw.context_tokens) ??
        (raw.step_count === 1 ? number(raw.prompt_tokens) : undefined);
      if (contextTokens === undefined) return null;
      const costUsd = number(raw.api_cost_usd);
      return {
        kind: "usage",
        usage: { contextTokens, ...(costUsd !== undefined ? { costUsd } : {}) },
      };
    }
    case "event_message": {
      const data = record(raw.event_data);
      if (raw.event_type === "compaction") {
        const trigger = typeof data?.trigger === "string" ? data.trigger : undefined;
        return { kind: "compaction_start", ...(trigger ? { trigger } : {}) };
      }
      if (raw.event_type === "retry" && data) {
        return {
          kind: "retry",
          retry: {
            attempt: number(data.attempt) ?? 0,
            maxAttempts: number(data.max_attempts) ?? 0,
            delayMs: number(data.delay_ms) ?? 0,
            ...(typeof data.message === "string" ? { reason: data.message } : {}),
          },
        };
      }
      return null;
    }
    case "summary_message": {
      const summary = typeof raw.summary === "string" ? raw.summary : "";
      const stats = record(raw.compaction_stats);
      const before = number(stats?.context_tokens_before);
      const after = number(stats?.context_tokens_after);
      return {
        kind: "compaction_summary",
        summary,
        ...(before !== undefined ? { contextTokensBefore: before } : {}),
        ...(after !== undefined ? { contextTokensAfter: after } : {}),
      };
    }
    default:
      return null;
  }
}

/** One line for a retry, e.g. "Retrying model request (2 of 3) in 5s: overloaded". */
export function retryText(retry: RetryReport): string {
  const attempt =
    retry.maxAttempts > 0
      ? ` (${retry.attempt} of ${retry.maxAttempts})`
      : retry.attempt > 0
        ? ` (attempt ${retry.attempt})`
        : "";
  const delay = retry.delayMs > 0 ? ` in ${formatSeconds(retry.delayMs)}` : "";
  const reason = retry.reason ? `: ${retry.reason}` : "";
  return `Retrying model request${attempt}${delay}${reason}`;
}

/** Summary text for a finished compaction, with the token change when known. */
export function compactionText(summary: CompactionSummary): string {
  const { contextTokensBefore: before, contextTokensAfter: after } = summary;
  const change =
    before !== undefined && after !== undefined
      ? `Context ${formatTokens(before)} → ${formatTokens(after)} tokens.`
      : "";
  return [change, summary.summary.trim()].filter(Boolean).join("\n\n");
}

function formatSeconds(ms: number): string {
  const seconds = ms / 1000;
  return `${Number.isInteger(seconds) ? seconds : seconds.toFixed(1)}s`;
}

function formatTokens(tokens: number): string {
  return tokens >= 1000 ? `${Math.round(tokens / 1000)}k` : String(tokens);
}

function number(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}
