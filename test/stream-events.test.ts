import { describe, expect, test } from "bun:test";
import { compactionText, parseStreamEvent, retryText } from "../src/stream-events.js";

describe("Letta stream events", () => {
  test("reads context usage and step cost", () => {
    expect(
      parseStreamEvent({
        message_type: "usage_statistics",
        prompt_tokens: 900,
        context_tokens: 1200,
        api_cost_usd: 0.004,
      }),
    ).toEqual({ kind: "usage", usage: { contextTokens: 1200, costUsd: 0.004 } });
    expect(
      parseStreamEvent({ message_type: "usage_statistics", prompt_tokens: 900 }),
    ).toEqual({ kind: "usage", usage: { contextTokens: 900 } });
    expect(parseStreamEvent({ message_type: "usage_statistics" })).toBeNull();
  });

  test("reads compaction start and summary", () => {
    expect(
      parseStreamEvent({
        message_type: "event_message",
        event_type: "compaction",
        event_data: { trigger: "context_window_limit" },
      }),
    ).toEqual({ kind: "compaction_start", trigger: "context_window_limit" });
    const summary = parseStreamEvent({
      message_type: "summary_message",
      summary: "Talked about tests.",
      compaction_stats: { context_tokens_before: 150_000, context_tokens_after: 12_400 },
    });
    expect(summary).toEqual({
      kind: "compaction_summary",
      summary: "Talked about tests.",
      contextTokensBefore: 150_000,
      contextTokensAfter: 12_400,
    });
    if (summary?.kind !== "compaction_summary") throw new Error("expected a summary");
    expect(compactionText(summary)).toBe(
      "Context 150k → 12k tokens.\n\nTalked about tests.",
    );
  });

  test("reads and describes retries", () => {
    const event = parseStreamEvent({
      message_type: "event_message",
      event_type: "retry",
      event_data: { attempt: 2, max_attempts: 3, delay_ms: 5000, message: "overloaded" },
    });
    expect(event).toEqual({
      kind: "retry",
      retry: { attempt: 2, maxAttempts: 3, delayMs: 5000, reason: "overloaded" },
    });
    if (event?.kind !== "retry") throw new Error("expected a retry");
    expect(retryText(event.retry)).toBe(
      "Retrying model request (2 of 3) in 5s: overloaded",
    );
  });
});
