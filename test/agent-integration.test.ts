import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentContext, McpServer } from "@agentclientprotocol/sdk";
import type {
  LettaCodeSocketConstructor,
  PermissionMode,
} from "@letta-ai/letta-agent-sdk";
import { LettaAcpAgent, toSdkMcpServers } from "../src/agent.js";
import { SessionRegistry } from "../src/session-registry.js";

interface RuntimeScope {
  agent_id: string;
  conversation_id: string;
}

type WireMessage = Record<string, unknown>;
type SocketListener = (event: unknown) => void;

interface ServerSocket {
  readonly channel: "control" | "stream";
  push(message: WireMessage): void;
}

class FakeAppServer {
  readonly runtimeStartRequestIds: string[] = [];
  readonly createdAgentBodies: WireMessage[] = [];
  readonly createdAgentPinGlobals: unknown[] = [];
  readonly approvalResponses: WireMessage[] = [];
  readonly conversationRetrieveIds: string[] = [];
  closedSockets = 0;
  readonly updatedModelPayloads: WireMessage[] = [];
  private nextConversation = 0;
  private activeControlSocket: ServerSocket | null = null;
  private activeRuntime: RuntimeScope | null = null;
  private activeCommand: string | null = null;
  private approvalWaiters: Array<(payload: WireMessage) => void> = [];
  private commandWaiters: Array<() => void> = [];
  private cancelablePromptWaiters: Array<() => void> = [];
  private cancelablePromptActive = false;
  private cancelableRunId: string | null = null;
  private replacementRuns = 0;
  private readonly scripts = new Map<
    string,
    { runId: string; deltas: WireMessage[]; afterStop: WireMessage[] }
  >();
  private abortSettling = false;

  socketConstructor(): LettaCodeSocketConstructor {
    const server = this;

    class FakeSocket {
      readyState = 0;
      readonly channel: "control" | "stream";
      private readonly listeners = new Map<string, Set<SocketListener>>();

      constructor(url: string) {
        this.channel = new URL(url).searchParams.get("channel") === "stream"
          ? "stream"
          : "control";
        queueMicrotask(() => {
          this.readyState = 1;
          this.emit("open", {});
        });
      }

      send(data: string): void {
        server.receive(this, JSON.parse(data) as WireMessage);
      }

      close(): void {
        if (this.readyState === 3) return;
        this.readyState = 3;
        server.closedSockets += 1;
        this.emit("close", {});
      }

      addEventListener(type: string, listener: SocketListener): void {
        const listeners = this.listeners.get(type) ?? new Set<SocketListener>();
        listeners.add(listener);
        this.listeners.set(type, listeners);
      }

      removeEventListener(type: string, listener: SocketListener): void {
        this.listeners.get(type)?.delete(listener);
      }

      push(message: WireMessage): void {
        queueMicrotask(() => {
          this.emit("message", { data: JSON.stringify(message) });
        });
      }

      private emit(type: string, event: unknown): void {
        for (const listener of this.listeners.get(type) ?? []) {
          listener(event);
        }
      }
    }

    return FakeSocket as unknown as LettaCodeSocketConstructor;
  }

  /**
   * Answers a prompt containing `trigger` with these stream deltas, then ends
   * the turn. `afterStop` deltas go between stop_reason and turn_finished,
   * where Letta sends usage_statistics.
   */
  script(
    trigger: string,
    runId: string,
    deltas: WireMessage[],
    afterStop: WireMessage[] = [],
  ): void {
    this.scripts.set(trigger, { runId, deltas, afterStop });
  }

  /** Resolves with the next approval_response the adapter sends back. */
  nextApprovalResponse(): Promise<WireMessage> {
    return new Promise((resolve) => {
      this.approvalWaiters.push(resolve);
    });
  }

  /** Resolves when the adapter dispatches its next execute_command. */
  nextCommand(): Promise<void> {
    return new Promise((resolve) => {
      this.commandWaiters.push(resolve);
    });
  }

  /** Resolves after a prompt has produced activity but before it completes. */
  nextCancelablePrompt(): Promise<void> {
    return new Promise((resolve) => {
      this.cancelablePromptWaiters.push(resolve);
    });
  }

  requestPermissionAfterPrompt(
    overrides: {
      requestId?: string;
      toolName?: string;
      toolCallId?: string;
      input?: Record<string, unknown>;
    } = {},
  ): void {
    const socket = this.activeControlSocket;
    const runtime = this.activeRuntime;
    if (!socket || !runtime) {
      throw new Error("No active runtime for late permission request");
    }
    socket.push({
      type: "control_request",
      request_id: overrides.requestId ?? "approval-request-after-prompt",
      runtime,
      request: {
        subtype: "can_use_tool",
        tool_name: overrides.toolName ?? "Bash",
        tool_call_id: overrides.toolCallId ?? "call-after-prompt",
        input: overrides.input ?? {
          command: "pwd",
          description: "Show working directory after prompt",
        },
      },
    });
  }

  private receive(socket: ServerSocket, message: WireMessage): void {
    if (socket.channel !== "control") {
      throw new Error(`Unexpected command on ${socket.channel} channel`);
    }

    switch (message.type) {
      case "runtime_start": {
        const requestId = requiredString(message.request_id, "runtime_start.request_id");
        const conversationId = `conv-test-${++this.nextConversation}`;
        const createAgent = message.create_agent as
          | { body?: WireMessage; pin_global?: boolean }
          | undefined;
        const agentId =
          typeof message.agent_id === "string"
            ? message.agent_id
            : createAgent
              ? "agent-created"
              : "agent-test";
        const runtime = {
          agent_id: agentId,
          conversation_id: conversationId,
        };
        this.runtimeStartRequestIds.push(requestId);
        if (createAgent?.body) {
          this.createdAgentBodies.push(createAgent.body);
          this.createdAgentPinGlobals.push(createAgent.pin_global);
        }
        socket.push({
          type: "runtime_start_response",
          request_id: requestId,
          success: true,
          runtime,
          agent: { id: agentId, model: "test/model", tools: [] },
          conversation: { id: conversationId, agent_id: agentId },
        });
        return;
      }
      case "enable_memfs":
        socket.push({
          type: "enable_memfs_response",
          request_id: requiredString(message.request_id, "enable_memfs.request_id"),
          success: true,
          memory_directory: "/tmp/letta-acp-memory",
        });
        return;
      case "conversation_retrieve": {
        const conversationId = requiredString(
          message.conversation_id,
          "conversation_retrieve.conversation_id",
        );
        this.conversationRetrieveIds.push(conversationId);
        if (conversationId.includes("missing")) {
          socket.push({
            type: "conversation_retrieve_response",
            request_id: requiredString(
              message.request_id,
              "conversation_retrieve.request_id",
            ),
            success: false,
            error: "conversation unavailable",
          });
          return;
        }
        socket.push({
          type: "conversation_retrieve_response",
          request_id: requiredString(
            message.request_id,
            "conversation_retrieve.request_id",
          ),
          success: true,
          conversation: {
            id: conversationId,
            agent_id: conversationId.includes("foreign") ? "agent-other" : "agent-test",
            archived: conversationId.includes("archived"),
          },
        });
        return;
      }
      case "conversation_messages_list":
        socket.push({
          type: "conversation_messages_list_response",
          request_id: requiredString(
            message.request_id,
            "conversation_messages_list.request_id",
          ),
          success: true,
          messages: [],
          has_more: false,
        });
        return;
      case "list_models":
        socket.push({
          type: "list_models_response",
          request_id: requiredString(message.request_id, "list_models.request_id"),
          success: true,
          entries: [
            {
              id: "test-model",
              handle: "test/model",
              label: "Test Model",
              description: "Deterministic test model",
              isDefault: true,
            },
            {
              id: "other-model",
              handle: "test/other-model",
              label: "Other Model",
              description: "Alternate deterministic model",
            },
            // One model offered at two reasoning tiers, as Letta lists them.
            {
              id: "tiered-low",
              handle: "test/tiered",
              label: "Tiered Model",
              description: "Model with reasoning tiers",
              updateArgs: { context_window: 200000, reasoning_effort: "low" },
            },
            {
              id: "tiered",
              handle: "test/tiered",
              label: "Tiered Model",
              description: "Model with reasoning tiers",
              updateArgs: { context_window: 200000, reasoning_effort: "high" },
            },
          ],
          available_handles: ["test/model", "test/other-model", "test/tiered"],
        });
        return;
      case "update_model": {
        const payload = message.payload as WireMessage;
        this.updatedModelPayloads.push(payload);
        const modelHandle =
          typeof payload.model_handle === "string" ? payload.model_handle : undefined;
        const modelId =
          typeof payload.model_id === "string"
            ? payload.model_id
            : modelHandle
              ? modelHandle.split("/").at(-1) ?? modelHandle
              : requiredString(payload.model_id, "update_model.model_id");
        const handle =
          modelHandle ??
          (modelId === "other-model"
            ? "test/other-model"
            : modelId.startsWith("tiered")
              ? "test/tiered"
              : "test/model");
        socket.push({
          type: "update_model_response",
          request_id: requiredString(message.request_id, "update_model.request_id"),
          success: true,
          runtime: message.runtime,
          applied_to: "conversation",
          model_id: modelId,
          model_handle: handle,
        });
        return;
      }
      case "input":
        this.handleInput(socket, message);
        return;
      case "execute_command":
        this.handleCommand(socket, message);
        return;
      case "abort_message":
        socket.push({
          type: "abort_message_response",
          request_id: requiredString(
            message.request_id,
            "abort_message.request_id",
          ),
          success: true,
        });
        if (this.activeCommand || this.cancelablePromptActive) {
          socket.push({
            type: "update_loop_status",
            runtime: this.activeRuntime,
            loop_status: {
              status: "WAITING_ON_INPUT",
              active_run_ids: [],
            },
          });
          this.cancelablePromptActive = false;
          // Letta Cloud settles the aborted run right after the idle status,
          // even when the run had produced no visible output yet.
          const run = this.cancelableRunId;
          if (run) {
            this.cancelableRunId = null;
            this.abortSettling = true;
            const runtime = this.activeRuntime!;
            setTimeout(() => {
              this.finishTurn(socket, runtime, run, "cancelled");
              this.abortSettling = false;
            }, 20);
          }
        }
        return;
      default:
        throw new Error(`Unhandled fake app-server command: ${String(message.type)}`);
    }
  }

  private handleInput(socket: ServerSocket, message: WireMessage): void {
    const runtime = message.runtime as RuntimeScope;
    const payload = message.payload as Record<string, unknown>;
    this.activeControlSocket = socket;
    this.activeRuntime = runtime;

    if (payload.kind === "approval_response") {
      this.approvalResponses.push(payload);
      for (const waiter of this.approvalWaiters.splice(0)) waiter(payload);
      const command = this.activeCommand;
      this.pushDelta(socket, runtime, {
        message_type: "tool_return_message",
        run_id: command ? "run-command-approval" : "run-approval",
        tool_call_id: command ? "call-command-approval" : "call-approval",
        tool_return: "write completed",
        status: "success",
      });
      this.pushDelta(socket, runtime, {
        message_type: "assistant_message",
        run_id: command ? "run-command-approval" : "run-approval",
        content: command ? "COMMAND_APPROVAL_OK" : "APPROVAL_OK",
      });
      socket.push({
        type: "update_loop_status",
        runtime,
        loop_status: {
          status: "WAITING_ON_INPUT",
          active_run_ids: [],
        },
      });
      if (command) {
        this.pushDelta(socket, runtime, {
          message_type: "slash_command_end",
          command_id: command,
          success: true,
          output: "COMMAND_APPROVAL_OK",
        });
        this.activeCommand = null;
      }
      return;
    }

    if (payload.kind !== "create_message") {
      throw new Error(`Unhandled fake input payload: ${String(payload.kind)}`);
    }

    if (this.abortSettling) {
      // Like Letta Cloud, finish processing the abort (and settle the aborted
      // run) before starting the next queued input.
      setTimeout(() => this.handleInput(socket, message), 40);
      return;
    }
    const promptText = JSON.stringify(payload.messages);
    if (promptText.includes("thinking prompt to cancel")) {
      // A run that has started (the model is thinking) but produced no
      // visible output yet, as when a user steers during a slow first step.
      this.cancelablePromptActive = true;
      this.cancelableRunId = "run-thinking";
      socket.push({
        type: "update_loop_status",
        runtime,
        loop_status: {
          status: "PROCESSING_API_RESPONSE",
          active_run_ids: ["run-thinking"],
        },
      });
      for (const waiter of this.cancelablePromptWaiters.splice(0)) waiter();
      return;
    }
    if (promptText.includes("silent prompt to cancel")) {
      // Accept the turn but emit nothing, as when a steer cancels a turn
      // before the model has produced its first event.
      this.cancelablePromptActive = true;
      for (const waiter of this.cancelablePromptWaiters.splice(0)) waiter();
      return;
    }
    if (promptText.includes("cancel active prompt")) {
      this.cancelablePromptActive = true;
      for (const waiter of this.cancelablePromptWaiters.splice(0)) waiter();
      this.pushDelta(socket, runtime, {
        message_type: "tool_call_message",
        run_id: "run-cancelled",
        tool_calls: [
          {
            id: "call-cancelled-agent",
            name: "Agent",
            arguments: JSON.stringify({
              description: "Inspect test coverage",
              prompt: "Review the repository tests",
            }),
          },
        ],
      });
      return;
    }
    if (promptText.includes("replacement after cancel")) {
      // Real runs have unique ids; the SDK ignores a repeated finished run.
      const runId = `run-replacement-${++this.replacementRuns}`;
      this.pushDelta(socket, runtime, {
        message_type: "assistant_message",
        run_id: runId,
        content: "REPLACEMENT_OK",
      });
      this.finishTurn(socket, runtime, runId, "end_turn");
      return;
    }
    for (const [trigger, script] of this.scripts) {
      if (!promptText.includes(trigger)) continue;
      for (const delta of script.deltas) {
        this.pushDelta(socket, runtime, { run_id: script.runId, ...delta });
      }
      this.pushDelta(socket, runtime, {
        message_type: "stop_reason",
        run_id: script.runId,
        stop_reason: "end_turn",
      });
      for (const delta of script.afterStop) {
        this.pushDelta(socket, runtime, { run_id: script.runId, ...delta });
      }
      socket.push({
        type: "turn_finished",
        runtime,
        run_id: script.runId,
        stop_reason: "end_turn",
      });
      return;
    }
    if (promptText.includes("fragmented prompt")) {
      this.streamFragmentedToolCall(socket, runtime);
      return;
    }
    if (promptText.includes("busy idle prompt")) {
      this.reportIdleWhileRunning(socket, runtime, { thenContinue: true });
      return;
    }
    if (promptText.includes("silent idle prompt")) {
      this.reportIdleWhileRunning(socket, runtime, { thenContinue: false });
      return;
    }

    const isApprovalPrompt = promptText.includes("approval prompt");
    if (isApprovalPrompt) {
      this.pushDelta(socket, runtime, {
        message_type: "tool_call_message",
        run_id: "run-approval",
        tool_calls: [
          {
            id: "call-approval",
            name: "Write",
            arguments: JSON.stringify({ file_path: "/tmp/test.txt" }),
          },
        ],
      });
      socket.push({
        type: "control_request",
        request_id: "approval-request-1",
        runtime,
        request: {
          subtype: "can_use_tool",
          tool_name: "Write",
          tool_call_id: "call-approval",
          input: { file_path: "/tmp/test.txt" },
        },
      });
      this.finishTurn(socket, runtime, "run-approval", "requires_approval");
      return;
    }

    if (promptText.includes("edit prompt")) {
      this.pushDelta(socket, runtime, {
        message_type: "tool_call_message",
        run_id: "run-edit",
        tool_calls: [
          {
            id: "call-edit",
            name: "Edit",
            arguments: JSON.stringify({
              file_path: "/tmp/example.ts",
              old_string: "const value = 1;",
              new_string: "const value = 2;",
            }),
          },
        ],
      });
      this.pushDelta(socket, runtime, {
        message_type: "tool_return_message",
        run_id: "run-edit",
        tool_call_id: "call-edit",
        tool_return: JSON.stringify({
          message: "Successfully replaced 1 occurrence",
          replacements: 1,
          startLine: 12,
        }),
        status: "success",
      });
      this.finishTurn(socket, runtime, "run-edit", "end_turn");
      return;
    }

    this.pushDelta(socket, runtime, {
      message_type: "assistant_message",
      run_id: "run-prompt",
      content: "PROMPT_OK",
    });
    this.finishTurn(socket, runtime, "run-prompt", "end_turn");
  }

  private handleCommand(socket: ServerSocket, message: WireMessage): void {
    const runtime = message.runtime as RuntimeScope;
    const command = requiredString(
      message.command_id,
      "execute_command.command_id",
    );
    this.activeControlSocket = socket;
    this.activeRuntime = runtime;
    this.activeCommand = command;
    for (const waiter of this.commandWaiters.splice(0)) waiter();

    this.pushDelta(socket, runtime, {
      message_type: "slash_command_start",
      command_id: command,
    });
    if (command === "init") {
      // Stay silent until the test cancels, reproducing an abort before the
      // command's nested turn emits its first substantive event.
      return;
    }
    this.pushDelta(socket, runtime, {
      message_type: "tool_call_message",
      run_id: "run-command-approval",
      tool_calls: [
        {
          id: "call-command-approval",
          name: "Write",
          arguments: JSON.stringify({ file_path: "/tmp/command-test.txt" }),
        },
      ],
    });
    socket.push({
      type: "control_request",
      request_id: "command-approval-request",
      runtime,
      request: {
        subtype: "can_use_tool",
        tool_name: "Write",
        tool_call_id: "call-command-approval",
        input: { file_path: "/tmp/command-test.txt" },
      },
    });
    this.finishTurn(socket, runtime, "run-command-approval", "requires_approval");
  }

  /**
   * Streams one tool call as the app-server really does: an opening chunk
   * carrying the name plus a slice of the arguments, then continuation chunks
   * that carry argument text only (and therefore no tool name).
   */
  private streamFragmentedToolCall(
    socket: ServerSocket,
    runtime: RuntimeScope,
  ): void {
    const fragments = [
      { name: "Bash", arguments: '{"command":"git status"' },
      { arguments: ',"description":"Show the ' },
      { arguments: 'working tree status"}' },
    ];
    for (const fragment of fragments) {
      this.pushDelta(socket, runtime, {
        message_type: "tool_call_message",
        run_id: "run-fragmented",
        tool_calls: [{ id: "call-fragmented", ...fragment }],
      });
    }
    this.pushDelta(socket, runtime, {
      message_type: "tool_return_message",
      run_id: "run-fragmented",
      tool_call_id: "call-fragmented",
      tool_return: "nothing to commit",
      status: "success",
    });
    this.finishTurn(socket, runtime, "run-fragmented", "end_turn");
  }

  /**
   * Reports WAITING_ON_INPUT while still listing an active run — the shape
   * that makes the SDK synthesize a successful, stop-reason-less result even
   * though the agent is mid-step. `thenContinue` decides whether the run
   * really was still going (after a beat, as a real server would take) or
   * whether the status was simply stale and nothing more arrives.
   */
  private reportIdleWhileRunning(
    socket: ServerSocket,
    runtime: RuntimeScope,
    options: { thenContinue: boolean },
  ): void {
    this.pushDelta(socket, runtime, {
      message_type: "assistant_message",
      run_id: "run-first",
      content: "WORKING",
    });
    socket.push({
      type: "update_loop_status",
      runtime,
      loop_status: {
        status: "WAITING_ON_INPUT",
        active_run_ids: ["run-second"],
      },
    });
    if (!options.thenContinue) return;
    // A real server takes a beat before the next step; without the delay the
    // continuation lands in the same microtask batch and hides the race.
    setTimeout(() => {
      this.pushDelta(socket, runtime, {
        message_type: "assistant_message",
        run_id: "run-second",
        content: "STILL_WORKING",
      });
      // The turn tracker was already consumed by the contradicted result, so
      // the run ends the way post-approval rounds do: on a real idle status.
      socket.push({
        type: "update_loop_status",
        runtime,
        loop_status: { status: "WAITING_ON_INPUT", active_run_ids: [] },
      });
    }, 20);
  }

  private pushDelta(
    socket: ServerSocket,
    runtime: RuntimeScope,
    delta: WireMessage,
  ): void {
    socket.push({ type: "stream_delta", runtime, delta });
  }

  /**
   * Agent SDK 0.8 parks a pending terminal on `stop_reason` and completes the
   * turn only after `turn_finished`. Letta Code has emitted that event since
   * 0.30.28; a fake that stops at `stop_reason` leaves the turn open until
   * `requestTimeoutMs`.
   */
  private finishTurn(
    socket: ServerSocket,
    runtime: RuntimeScope,
    runId: string,
    stopReason: string,
  ): void {
    this.pushDelta(socket, runtime, {
      message_type: "stop_reason",
      run_id: runId,
      stop_reason: stopReason,
    });
    socket.push({
      type: "turn_finished",
      runtime,
      run_id: runId,
      stop_reason: stopReason,
    });
  }
}

/** Fails instead of hanging when a prompt never settles. */
async function raceTimeout<T>(pending: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      pending,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`timed out after ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Missing ${label}`);
  }
  return value;
}

function createAgent(
  server: FakeAppServer,
  overrides: {
    agentId?: string;
    model?: string;
    sessionRegistryDir?: string | null;
    sessionRegistryScope?: string;
    outOfTurnPermissionTimeoutMs?: number;
    prematureResultGraceMs?: number;
    permissionSelector?: boolean;
    cancelDrainMs?: number;
    permissionMode?: PermissionMode;
  } = {},
): LettaAcpAgent {
  return new LettaAcpAgent({
    clientOptions: {
      backend: "local",
      appServer: {
        url: "ws://fake-app-server.test/ws",
        WebSocket: server.socketConstructor(),
        requestTimeoutMs: 1_000,
        pinGlobalAgent: false,
      },
    },
    agentId: "agent-test",
    permissionMode: "standard",
    ...overrides,
  });
}

function createContext(
  options: {
    answerPermissions?: boolean;
    permissionDelayMs?: number;
    permissionOptionId?: "allow_once" | "allow_always" | "reject_once";
  } = {},
) {
  const answerPermissions = options.answerPermissions ?? true;
  const permissionDelayMs = options.permissionDelayMs ?? 0;
  const permissionOptionId = options.permissionOptionId ?? "allow_once";
  const updates: WireMessage[] = [];
  const permissionRequests: WireMessage[] = [];
  const permissionSignals: Array<AbortSignal | undefined> = [];
  const context = {
    notify: async (_method: unknown, params: { update: WireMessage }) => {
      updates.push(params.update);
    },
    request: async (
      _method: unknown,
      params: WireMessage,
      requestOptions?: { cancellationSignal?: AbortSignal },
    ) => {
      permissionRequests.push(params);
      permissionSignals.push(requestOptions?.cancellationSignal);
      if (!answerPermissions) return new Promise(() => {});
      if (permissionDelayMs > 0) {
        await Bun.sleep(permissionDelayMs);
      }
      return {
        outcome: { outcome: "selected" as const, optionId: permissionOptionId },
      };
    },
  } as unknown as AgentContext;
  return { context, updates, permissionRequests, permissionSignals };
}

async function openSession(
  agent: LettaAcpAgent,
  context: AgentContext,
  cwd = "/tmp/letta-acp-test",
  clientCapabilities: Record<string, unknown> = {},
) {
  await agent.initialize({ protocolVersion: 1, clientCapabilities });
  return agent.newSession({ cwd, mcpServers: [] }, context);
}

describe("Agent SDK app-server integration", () => {
  test("advertises and maps Agent SDK MCP transports", async () => {
    const servers: McpServer[] = [
      {
        name: "local",
        command: "/usr/bin/node",
        args: ["server.js"],
        env: [{ name: "TOKEN", value: "local-token" }],
      },
      {
        type: "http",
        name: "remote",
        url: "https://example.com/mcp",
        headers: [{ name: "Authorization", value: "Bearer token" }],
      },
      {
        type: "sse",
        name: "legacy",
        url: "https://example.com/sse",
        headers: [],
      },
      {
        type: "acp",
        name: "in-band",
        serverId: "server-1",
      },
    ];

    expect(toSdkMcpServers(servers)).toEqual({
      local: {
        command: "/usr/bin/node",
        args: ["server.js"],
        env: { TOKEN: "local-token" },
      },
      remote: {
        type: "http",
        url: "https://example.com/mcp",
        headers: { Authorization: "Bearer token" },
      },
      legacy: {
        type: "sse",
        url: "https://example.com/sse",
        headers: {},
      },
    });

    const agent = createAgent(new FakeAppServer());
    try {
      const response = await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
      });
      expect(response.agentCapabilities?.mcpCapabilities).toEqual({
        http: true,
        sse: true,
      });
    } finally {
      agent.shutdown();
    }
  });

  test("creates a named, unpinned ACP agent with the configured model", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { agentId: undefined });
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);

      expect(session.sessionId).toBe("conv-test-2");
      expect(server.createdAgentBodies).toHaveLength(1);
      expect(server.createdAgentBodies[0]).toMatchObject({
        name: "Letta",
        description: "Letta agent driven through the Agent Client Protocol",
        model: "letta/auto",
        memory_blocks: expect.arrayContaining([
          expect.objectContaining({ label: "persona" }),
          expect.objectContaining({ label: "human" }),
        ]),
      });
      expect(server.createdAgentBodies[0]).not.toHaveProperty("hidden", true);
      expect(server.createdAgentPinGlobals).toEqual([false]);
    } finally {
      agent.shutdown();
    }
  });

  test("uses an explicit model when creating an agent", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, {
      agentId: undefined,
      model: "anthropic/claude-sonnet-4",
    });
    const { context } = createContext();

    try {
      await openSession(agent, context);
      expect(server.createdAgentBodies[0]).toMatchObject({
        model: "anthropic/claude-sonnet-4",
      });
      expect(server.createdAgentPinGlobals).toEqual([false]);
    } finally {
      agent.shutdown();
    }
  });

  test("reuses an explicit agent without creating or pinning another", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { agentId: "agent-existing" });
    const { context } = createContext();

    try {
      await openSession(agent, context);
      expect(server.createdAgentBodies).toEqual([]);
      expect(server.createdAgentPinGlobals).toEqual([]);
      expect(server.updatedModelPayloads).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("closes a displaced session when the same conversation is loaded again", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();
    const params = {
      sessionId: "conv-existing",
      cwd: "/tmp/letta-acp-test",
      mcpServers: [],
    };

    try {
      await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
      await agent.loadSession(params, context);
      const closedBeforeReload = server.closedSockets;

      await agent.loadSession(params, context);

      expect(server.closedSockets).toBeGreaterThan(closedBeforeReload);
    } finally {
      agent.shutdown();
    }
  });

  test("bounds session discovery lookups to one registry page", async () => {
    const directory = await mkdtemp(join(tmpdir(), "letta-acp-list-page-"));
    const scope = "pagination-test";
    const registry = new SessionRegistry(directory, scope);
    const cwd = "/tmp/letta-acp-test";
    for (let index = 0; index < 51; index += 1) {
      await registry.record("agent-test", `conv-page-${String(index).padStart(2, "0")}`, cwd);
    }

    const server = new FakeAppServer();
    const agent = createAgent(server, {
      sessionRegistryDir: directory,
      sessionRegistryScope: scope,
    });
    try {
      const first = await agent.listSessions({ cwd });
      expect(first.sessions).toHaveLength(50);
      expect(first.nextCursor).toBeString();
      expect(server.conversationRetrieveIds).toHaveLength(50);

      const second = await agent.listSessions({ cwd, cursor: first.nextCursor });
      expect(second.sessions).toHaveLength(1);
      expect(second.nextCursor).toBeUndefined();
      expect(server.conversationRetrieveIds).toHaveLength(51);
    } finally {
      agent.shutdown();
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("removes archived and foreign sessions but retains unavailable records", async () => {
    const directory = await mkdtemp(join(tmpdir(), "letta-acp-list-stale-"));
    const scope = "stale-test";
    const registry = new SessionRegistry(directory, scope);
    const cwd = "/tmp/letta-acp-test";
    await registry.record("agent-test", "conv-visible", cwd);
    await registry.record("agent-test", "conv-archived", cwd);
    await registry.record("agent-test", "conv-foreign", cwd);
    await registry.record("agent-test", "conv-missing", cwd);

    const server = new FakeAppServer();
    const agent = createAgent(server, {
      sessionRegistryDir: directory,
      sessionRegistryScope: scope,
    });
    try {
      const result = await agent.listSessions({ cwd });
      expect(result.sessions.map((session) => session.sessionId)).toEqual([
        "conv-visible",
      ]);
      expect(
        (await registry.list("agent-test", cwd)).map((record) => record.sessionId).sort(),
      ).toEqual(["conv-missing", "conv-visible"]);
    } finally {
      agent.shutdown();
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("starts repeated sessions with distinct runtime request ids", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const sessionIds: string[] = [];
      for (let index = 0; index < 3; index += 1) {
        sessionIds.push((await openSession(agent, context)).sessionId);
      }

      expect(sessionIds).toEqual([
        "conv-test-1",
        "conv-test-2",
        "conv-test-3",
      ]);
      expect(new Set(server.runtimeStartRequestIds).size).toBe(3);
    } finally {
      agent.shutdown();
    }
  });

  test("advertises available models in the session configuration", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);

      expect(session.configOptions).toEqual([
        {
          id: "permissions",
          name: "Permissions",
          description: "Approval behavior for tool calls",
          category: "mode",
          type: "select",
          currentValue: "standard",
          options: [
            expect.objectContaining({ value: "standard", name: "Ask before edits" }),
            expect.objectContaining({ value: "acceptEdits", name: "Accept edits" }),
            expect.objectContaining({
              value: "unrestricted",
              name: "Bypass permissions",
            }),
          ],
        },
        {
          id: "model",
          name: "Model",
          description: "Model used for this Letta session",
          category: "model",
          type: "select",
          currentValue: "test-model",
          options: [
            {
              value: "test-model",
              name: "Test Model",
              description: "Deterministic test model",
            },
            {
              value: "other-model",
              name: "Other Model",
              description: "Alternate deterministic model",
            },
            {
              value: "tiered",
              name: "Tiered Model",
              description: "Model with reasoning tiers",
            },
          ],
        },
        {
          id: "reasoning_effort",
          name: "Reasoning",
          description: "This model has no reasoning effort settings",
          category: "thought_level",
          type: "select",
          currentValue: "default",
          options: [{ value: "default", name: "Default" }],
        },
      ]);
      // Legacy model state carries each model's tiers for per-model pickers.
      expect(
        (session as { models?: unknown }).models,
      ).toEqual({
        currentModelId: "test-model",
        availableModels: [
          {
            modelId: "test-model",
            name: "Test Model",
            description: "Deterministic test model",
            _meta: { supportsReasoningEffort: false },
          },
          {
            modelId: "other-model",
            name: "Other Model",
            description: "Alternate deterministic model",
            _meta: { supportsReasoningEffort: false },
          },
          {
            modelId: "tiered",
            name: "Tiered Model",
            description: "Model with reasoning tiers",
            _meta: {
              supportsReasoningEffort: true,
              reasoningEffort: "high",
              reasoningEfforts: [
                { value: "low", label: "Low" },
                { value: "high", label: "High" },
              ],
            },
          },
        ],
      });
    } finally {
      agent.shutdown();
    }
  });

  test("hides the permission selector when disabled", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { permissionSelector: false });
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);
      expect(session.modes).toBeUndefined();
      expect(session.configOptions?.map((option) => option.id)).toEqual([
        "model",
        "reasoning_effort",
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("switches models through session configuration options", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "model",
        value: "other-model",
      });

      expect(server.updatedModelPayloads).toEqual([{ model_id: "other-model" }]);
      expect(
        result.configOptions.find((option) => option.id === "model"),
      ).toMatchObject({
        id: "model",
        category: "model",
        currentValue: "other-model",
      });
    } finally {
      agent.shutdown();
    }
  });

  test("selects reasoning effort separately from the model", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);
      const tiered = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "model",
        value: "tiered",
      });
      expect(server.updatedModelPayloads.at(-1)).toEqual({ model_id: "tiered" });
      expect(tiered.configOptions.map((option) => option.id)).toEqual([
        "permissions",
        "model",
        "reasoning_effort",
      ]);
      expect(
        tiered.configOptions.find((option) => option.id === "reasoning_effort"),
      ).toMatchObject({
        category: "thought_level",
        currentValue: "high",
        options: [
          { value: "low", name: "Low" },
          { value: "high", name: "High" },
        ],
      });

      const low = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "reasoning_effort",
        value: "low",
      });
      expect(server.updatedModelPayloads.at(-1)).toEqual({ model_id: "tiered-low" });
      expect(
        low.configOptions.find((option) => option.id === "model"),
      ).toMatchObject({ currentValue: "tiered" });
      expect(
        low.configOptions.find((option) => option.id === "reasoning_effort"),
      ).toMatchObject({ currentValue: "low" });

      // An untiered model narrows the effort selector to "Default".
      const other = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "model",
        value: "other-model",
      });
      expect(
        other.configOptions.find((option) => option.id === "reasoning_effort"),
      ).toMatchObject({
        currentValue: "default",
        options: [{ value: "default", name: "Default" }],
      });
      const payloadCount = server.updatedModelPayloads.length;
      await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "reasoning_effort",
        value: "default",
      });
      expect(server.updatedModelPayloads).toHaveLength(payloadCount);

      await expect(
        agent.setSessionConfigOption({
          sessionId: session.sessionId,
          configId: "reasoning_effort",
          value: "low",
        }),
      ).rejects.toThrow("has no low reasoning effort");
    } finally {
      agent.shutdown();
    }
  });

  test("switches permission enforcement through session config options", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "permissions",
        value: "unrestricted",
      });
      expect(
        result.configOptions.find((option) => option.id === "permissions"),
      ).toMatchObject({
        category: "mode",
        currentValue: "unrestricted",
      });

      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt();

      expect(await approval).toMatchObject({
        decision: { behavior: "allow" },
      });
      expect(permissionRequests).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("streams an assistant response through a complete prompt turn", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(updates).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "PROMPT_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("streams fragmented tool arguments as one tool card", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "fragmented prompt" }],
        },
        context,
      );
      expect(result).toEqual({ stopReason: "end_turn" });

      expect(
        updates.filter((update) => update.toolCallId === "call-fragmented"),
      ).toEqual([
        {
          sessionUpdate: "tool_call",
          toolCallId: "call-fragmented",
          title: "Bash",
          kind: "execute",
          status: "in_progress",
          rawInput: { raw: '{"command":"git status"' },
          locations: [],
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          title: "Show the working tree status",
          kind: "execute",
          status: "in_progress",
          rawInput: {
            command: "git status",
            description: "Show the working tree status",
          },
          locations: [],
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          status: "completed",
          rawOutput: "nothing to commit",
          content: [
            {
              type: "content",
              content: { type: "text", text: "nothing to commit" },
            },
          ],
        },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("renders Bash output as a native terminal when the client supports it", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: { _meta: { terminal_output: true } },
      });
      const session = await agent.newSession(
        { cwd: "/tmp/letta-acp-test", mcpServers: [] },
        context,
      );
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "fragmented prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(
        updates.filter((update) => update.toolCallId === "call-fragmented"),
      ).toEqual([
        {
          sessionUpdate: "tool_call",
          toolCallId: "call-fragmented",
          title: "Bash",
          kind: "execute",
          status: "in_progress",
          rawInput: { raw: '{"command":"git status"' },
          locations: [],
          content: [{ type: "terminal", terminalId: "call-fragmented" }],
          _meta: {
            terminal_info: {
              terminal_id: "call-fragmented",
              cwd: "/tmp/letta-acp-test",
            },
          },
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          title: "Show the working tree status",
          kind: "execute",
          status: "in_progress",
          rawInput: {
            command: "git status",
            description: "Show the working tree status",
          },
          locations: [],
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          _meta: {
            terminal_output: {
              terminal_id: "call-fragmented",
              data: "nothing to commit",
            },
          },
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "call-fragmented",
          status: "completed",
          rawOutput: "nothing to commit",
          content: [
            { type: "terminal", terminalId: "call-fragmented" },
            {
              type: "content",
              content: { type: "text", text: "nothing to commit" },
            },
          ],
          _meta: {
            terminal_exit: {
              terminal_id: "call-fragmented",
              exit_code: 0,
              signal: null,
            },
          },
        },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("renders an SDK Edit call as a native ACP diff", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "edit prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(updates).toContainEqual(
        expect.objectContaining({
          sessionUpdate: "tool_call",
          toolCallId: "call-edit",
          content: [
            {
              type: "diff",
              path: "/tmp/example.ts",
              oldText: "const value = 1;",
              newText: "const value = 2;",
            },
          ],
        }),
      );
      const completed = updates.find(
        (update) =>
          update.sessionUpdate === "tool_call_update" &&
          update.toolCallId === "call-edit" &&
          update.status === "completed",
      );
      expect(completed).toEqual(
        expect.objectContaining({
          rawOutput: {
            message: "Successfully replaced 1 occurrence",
            replacements: 1,
            startLine: 12,
          },
          locations: [{ path: "/tmp/example.ts", line: 12 }],
        }),
      );
      expect(completed).not.toHaveProperty("content");
    } finally {
      agent.shutdown();
    }
  });

  test("routes a permission request after the prompt handler returns", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests, permissionSignals } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      expect(result).toEqual({ stopReason: "end_turn" });

      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt();

      expect(await approval).toEqual(
        expect.objectContaining({
          kind: "approval_response",
          request_id: "approval-request-after-prompt",
          decision: expect.objectContaining({ behavior: "allow" }),
        }),
      );
      expect(permissionRequests).toContainEqual(
        expect.objectContaining({
          sessionId: session.sessionId,
          toolCall: expect.objectContaining({
            title: "Show working directory after prompt",
          }),
        }),
      );
      // Out-of-turn requests carry a deadline the client can observe.
      expect(permissionSignals.at(-1)).toBeInstanceOf(AbortSignal);
    } finally {
      agent.shutdown();
    }
  });

  test("reports the task list as an ACP plan", async () => {
    const server = new FakeAppServer();
    server.script("plan prompt", "run-plan", [
      {
        message_type: "tool_call_message",
        tool_calls: [
          { id: "call-create", name: "TaskCreate", arguments: JSON.stringify({ subject: "Run tests" }) },
        ],
      },
      {
        message_type: "tool_return_message",
        tool_call_id: "call-create",
        tool_return: JSON.stringify({ taskId: "task_1", subject: "Run tests", status: "pending" }),
        status: "success",
      },
      {
        message_type: "tool_call_message",
        tool_calls: [
          {
            id: "call-update",
            name: "TaskUpdate",
            arguments: JSON.stringify({ taskId: "task_1", status: "completed" }),
          },
        ],
      },
      {
        message_type: "tool_return_message",
        tool_call_id: "call-update",
        tool_return: JSON.stringify({ taskId: "task_1", subject: "Run tests", status: "completed" }),
        status: "success",
      },
    ]);
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        { sessionId: session.sessionId, prompt: [{ type: "text", text: "plan prompt" }] },
        context,
      );
      expect(updates.filter((update) => update.sessionUpdate === "plan")).toEqual([
        {
          sessionUpdate: "plan",
          entries: [{ content: "Run tests", priority: "medium", status: "pending" }],
        },
        {
          sessionUpdate: "plan",
          entries: [{ content: "Run tests", priority: "medium", status: "completed" }],
        },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("reports context usage, cumulative cost, compaction and retries", async () => {
    const server = new FakeAppServer();
    server.script(
      "usage prompt",
      "run-usage",
      [
        {
          message_type: "event_message",
          event_type: "retry",
          event_data: { attempt: 1, max_attempts: 3, delay_ms: 2000, message: "overloaded" },
        },
        {
          message_type: "event_message",
          event_type: "compaction",
          event_data: { trigger: "context_window_limit", context_window: 40_000 },
        },
        {
          message_type: "summary_message",
          summary: "Earlier work summarized.",
          compaction_stats: { context_tokens_before: 90_000, context_tokens_after: 8_000 },
        },
        { message_type: "assistant_message", content: "USAGE_OK" },
        { message_type: "usage_statistics", context_tokens: 8_500, api_cost_usd: 0.01 },
      ],
      [{ message_type: "usage_statistics", context_tokens: 9_000, api_cost_usd: 0.02 }],
    );
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        { sessionId: session.sessionId, prompt: [{ type: "text", text: "usage prompt" }] },
        context,
      );
      const usage = updates.filter((update) => update.sessionUpdate === "usage_update");
      expect(usage.map((update) => update.used)).toEqual([8_500, 9_000]);
      // The window the server reported with the compaction wins over the catalog's.
      expect(usage.map((update) => update.size)).toEqual([40_000, 40_000]);
      expect((usage.at(-1)?.cost as { amount: number }).amount).toBeCloseTo(0.03);
      expect(
        updates.find((update) => update.sessionUpdate === "agent_thought_chunk"),
      ).toEqual({
        sessionUpdate: "agent_thought_chunk",
        content: {
          type: "text",
          text: "\nRetrying model request (1 of 3) in 2s: overloaded\n",
        },
      });
      // Without the session.compaction capability, compaction is a tool card.
      const compaction = updates.filter(
        (update) => update.toolCallId === "compaction-1",
      );
      expect(compaction).toEqual([
        {
          sessionUpdate: "tool_call",
          toolCallId: "compaction-1",
          title: "Compact context",
          kind: "think",
          status: "in_progress",
        },
        {
          sessionUpdate: "tool_call_update",
          toolCallId: "compaction-1",
          status: "completed",
          content: [
            {
              type: "content",
              content: {
                type: "text",
                text: "Context 90k → 8k tokens.\n\nEarlier work summarized.",
              },
            },
          ],
        },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("closes a compaction card when the server sends no summary", async () => {
    const server = new FakeAppServer();
    server.script("silent compaction prompt", "run-silent-compaction", [
      {
        message_type: "event_message",
        event_type: "compaction",
        event_data: { trigger: "context_window_limit" },
      },
      { message_type: "assistant_message", content: "AFTER_COMPACTION" },
    ]);
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "silent compaction prompt" }],
        },
        context,
      );
      const compaction = updates.filter((update) => update.toolCallId === "compaction-1");
      expect(compaction.map((update) => update.status)).toEqual(["in_progress", "completed"]);
      // The card closes before the reply that follows it.
      const closed = updates.findIndex(
        (update) => update.toolCallId === "compaction-1" && update.status === "completed",
      );
      const reply = updates.findIndex(
        (update) => update.sessionUpdate === "agent_message_chunk",
      );
      expect(closed).toBeLessThan(reply);
    } finally {
      agent.shutdown();
    }
  });

  test("uses native compaction updates and notices when the client has them", async () => {
    const server = new FakeAppServer();
    server.script("native compaction prompt", "run-native", [
      {
        message_type: "event_message",
        event_type: "retry",
        event_data: { attempt: 2, max_attempts: 3, delay_ms: 500 },
      },
      { message_type: "summary_message", summary: "Short." },
    ]);
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context, undefined, {
        session: { compaction: {}, notices: {} },
      });
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "native compaction prompt" }],
        },
        context,
      );
      expect(updates).toContainEqual({
        sessionUpdate: "notice",
        severity: "warning",
        title: "Retrying model request (2 of 3) in 0.5s",
      });
      expect(
        updates.filter((update) => update.sessionUpdate === "compaction_update"),
      ).toEqual([
        { sessionUpdate: "compaction_update", compactionId: "compaction-1", status: "in_progress" },
        {
          sessionUpdate: "compaction_update",
          compactionId: "compaction-1",
          status: "completed",
          summary: [{ type: "text", text: "Short." }],
        },
      ]);
    } finally {
      agent.shutdown();
    }
  });

  test("auto-allows task bookkeeping in ask-before-edits mode", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "task-update-request",
        toolName: "TaskUpdate",
        toolCallId: "call-task-update",
        input: { taskId: "task_1", status: "completed" },
      });

      expect(await approval).toEqual(
        expect.objectContaining({
          request_id: "task-update-request",
          decision: expect.objectContaining({ behavior: "allow" }),
        }),
      );
      expect(permissionRequests).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("aligns editor-tool approvals with session modes and cwd", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();
    const root = await mkdtemp(join(tmpdir(), "letta-acp-editor-permissions-"));
    const workspace = join(root, "repo");
    const inside = join(workspace, "src", "inside.ts");
    const outside = join(root, "outside.ts");
    await mkdir(join(workspace, "src"), { recursive: true });
    await writeFile(inside, "inside\n");
    await writeFile(outside, "outside\n");

    try {
      const session = await openSession(agent, context, workspace);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );

      const insideApproval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "read-inside",
        toolName: "read_editor_buffer",
        toolCallId: "call-read-inside",
        input: { path: inside },
      });
      expect(await insideApproval).toMatchObject({
        request_id: "read-inside",
        decision: { behavior: "allow" },
      });
      expect(permissionRequests).toEqual([]);

      const outsideApproval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "read-outside",
        toolName: "read_editor_buffer",
        toolCallId: "call-read-outside",
        input: { path: outside },
      });
      await outsideApproval;
      expect(permissionRequests.at(-1)).toMatchObject({
        toolCall: {
          toolCallId: "call-read-outside",
          kind: "read",
          rawInput: { path: outside },
        },
      });

      const standardWrite = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "write-standard",
        toolName: "write_via_editor",
        toolCallId: "call-write-standard",
        input: { path: inside, content: "changed\n" },
      });
      await standardWrite;
      expect(permissionRequests.at(-1)).toMatchObject({
        toolCall: {
          toolCallId: "call-write-standard",
          kind: "edit",
        },
      });

      await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "permissions",
        value: "acceptEdits",
      });
      const requestCount = permissionRequests.length;
      const acceptedWrite = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "write-accepted",
        toolName: "write_via_editor",
        toolCallId: "call-write-accepted",
        input: { path: inside, content: "accepted\n" },
      });
      expect(await acceptedWrite).toMatchObject({
        request_id: "write-accepted",
        decision: { behavior: "allow" },
      });
      expect(permissionRequests).toHaveLength(requestCount);
    } finally {
      agent.shutdown();
      await rm(root, { recursive: true, force: true });
    }
  });

  test("scopes always-allow editor decisions to one ACP session", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext({
      permissionOptionId: "allow_always",
    });
    const root = await mkdtemp(join(tmpdir(), "letta-acp-editor-always-"));
    const firstWorkspace = join(root, "first");
    const secondWorkspace = join(root, "second");
    const outside = join(root, "outside.ts");
    await mkdir(firstWorkspace, { recursive: true });
    await mkdir(secondWorkspace, { recursive: true });
    await writeFile(outside, "outside\n");

    try {
      const first = await openSession(agent, context, firstWorkspace);
      await agent.prompt(
        {
          sessionId: first.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      for (const requestId of ["first-read", "first-read-again"]) {
        const approval = server.nextApprovalResponse();
        server.requestPermissionAfterPrompt({
          requestId,
          toolName: "read_editor_buffer",
          toolCallId: `call-${requestId}`,
          input: { path: outside },
        });
        await approval;
      }
      expect(permissionRequests).toHaveLength(1);

      const second = await openSession(agent, context, secondWorkspace);
      await agent.prompt(
        {
          sessionId: second.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      const secondApproval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "second-read",
        toolName: "read_editor_buffer",
        toolCallId: "call-second-read",
        input: { path: outside },
      });
      await secondApproval;
      expect(permissionRequests).toHaveLength(2);
    } finally {
      agent.shutdown();
      await rm(root, { recursive: true, force: true });
    }
  });

  test("attaches a permission request to the tool call it belongs to", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      // Leaves a streamed Bash call as the most recent one, so name matching
      // cannot find the Write call the approval is actually for.
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "fragmented prompt" }],
        },
        context,
      );

      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt({
        requestId: "approval-unrelated-tool",
        toolName: "Write",
        toolCallId: "call-write",
        input: { file_path: "/tmp/example.ts" },
      });
      await approval;

      // A synthetic id here would render a permission card the client can
      // never reconcile with the tool call in the stream.
      expect(permissionRequests.at(-1)).toEqual(
        expect.objectContaining({
          toolCall: expect.objectContaining({
            toolCallId: "call-write",
            title: "Write: /tmp/example.ts",
            kind: "edit",
          }),
        }),
      );
    } finally {
      agent.shutdown();
    }
  });

  test("denies a permission request that lands after cancellation", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );
      await agent.cancel({ sessionId: session.sessionId });

      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt();

      expect(await approval).toEqual(
        expect.objectContaining({
          request_id: "approval-request-after-prompt",
          decision: expect.objectContaining({
            behavior: "deny",
            message: "Prompt turn was cancelled",
          }),
        }),
      );
      // The user already pressed stop; they should not be asked again.
      expect(permissionRequests).toEqual([]);
    } finally {
      agent.shutdown();
    }
  });

  test("denies an out-of-turn permission request the client never answers", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { outOfTurnPermissionTimeoutMs: 10 });
    const { context, permissionSignals } = createContext({
      answerPermissions: false,
    });

    try {
      const session = await openSession(agent, context);
      await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "plain prompt" }],
        },
        context,
      );

      const approval = server.nextApprovalResponse();
      server.requestPermissionAfterPrompt();

      expect(await approval).toEqual(
        expect.objectContaining({
          request_id: "approval-request-after-prompt",
          decision: expect.objectContaining({
            behavior: "deny",
            message:
              "Permission request timed out after 10ms while trying to " +
              "approve Bash outside a prompt turn",
          }),
        }),
      );
      expect(permissionSignals.at(-1)?.aborted).toBe(true);
    } finally {
      agent.shutdown();
    }
  });

  test("ends a contradicted turn when the SDK drops the still-running continuation", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "busy idle prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      // SDK 0.8 settles every run id listed on the contradicted idle status.
      // The continuation for that run is dropped before this adapter can see
      // it, so the prompt must end instead of waiting for a second idle.
      expect(updates).not.toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "STILL_WORKING" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("ends the turn when a contradicted completion is never followed up", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { prematureResultGraceMs: 20 });
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);
      // The status was simply stale: nothing follows it. The grace period has
      // to expire on its own, or the prompt would hang forever.
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "silent idle prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
    } finally {
      agent.shutdown();
    }
  });

  test("recovers a prompt after an approved mutating tool call", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates, permissionRequests } = createContext();

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "approval prompt" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(permissionRequests).toHaveLength(1);
      expect(permissionRequests[0]).toMatchObject({
        sessionId: session.sessionId,
        toolCall: { title: "Write: /tmp/test.txt" },
      });
      expect(server.approvalResponses).toHaveLength(1);
      expect(server.approvalResponses[0]).toMatchObject({
        kind: "approval_response",
        request_id: "approval-request-1",
        decision: { behavior: "allow" },
      });
      expect(updates).toContainEqual(
        expect.objectContaining({
          sessionUpdate: "tool_call_update",
          toolCallId: "call-approval",
          status: "completed",
          content: [
            {
              type: "content",
              content: { type: "text", text: "write completed" },
            },
          ],
        }),
      );
      expect(updates).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "APPROVAL_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("recovers an execute command after an approved tool call", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates, permissionRequests } = createContext({
      permissionDelayMs: 10,
    });

    try {
      const session = await openSession(agent, context);
      const result = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "/remember command fact" }],
        },
        context,
      );

      expect(result).toEqual({ stopReason: "end_turn" });
      expect(permissionRequests).toHaveLength(1);
      expect(permissionRequests[0]).toMatchObject({
        sessionId: session.sessionId,
        toolCall: {
          toolCallId: "call-command-approval",
          title: "Write: /tmp/command-test.txt",
        },
      });
      expect(updates).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "COMMAND_APPROVAL_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("drains a cancelled turn before starting its replacement", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const started = server.nextCancelablePrompt();
      const first = agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "cancel active prompt" }],
        },
        context,
      );
      await started;
      await agent.cancel({ sessionId: session.sessionId });
      expect(await first).toEqual({ stopReason: "cancelled" });

      const replacementUpdateStart = updates.length;
      const replacement = await agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "replacement after cancel" }],
        },
        context,
      );

      expect(replacement).toEqual({ stopReason: "end_turn" });
      expect(updates.slice(replacementUpdateStart)).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "REPLACEMENT_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("a replacement prompt gets its own turn after an early cancel", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const started = server.nextCancelablePrompt();
      const first = agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "silent prompt to cancel" }],
        },
        context,
      );
      await started;
      // Steering: the client cancels, then immediately sends the new prompt.
      await agent.cancel({ sessionId: session.sessionId });
      expect(await first).toEqual({ stopReason: "cancelled" });

      const replacementUpdateStart = updates.length;
      const replacement = await raceTimeout(
        agent.prompt(
          {
            sessionId: session.sessionId,
            prompt: [{ type: "text", text: "replacement after cancel" }],
          },
          context,
        ),
        3000,
      );

      expect(replacement).toEqual({ stopReason: "end_turn" });
      expect(updates.slice(replacementUpdateStart)).toContainEqual({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "REPLACEMENT_OK" },
      });
    } finally {
      agent.shutdown();
    }
  });

  test("a steer during a silent first step keeps later replies in order", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server, { cancelDrainMs: 1_000 });
    const { context, updates } = createContext();

    try {
      const session = await openSession(agent, context);
      const started = server.nextCancelablePrompt();
      const first = agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "thinking prompt to cancel" }],
        },
        context,
      );
      await started;
      await agent.cancel({ sessionId: session.sessionId });
      expect(await first).toEqual({ stopReason: "cancelled" });

      // The cancelled run's late "cancelled" result must not end (or fail)
      // the steered prompt, and each later prompt must get its own reply.
      for (const _ of [1, 2]) {
        const start = updates.length;
        const reply = await raceTimeout(
          agent.prompt(
            {
              sessionId: session.sessionId,
              prompt: [{ type: "text", text: "replacement after cancel" }],
            },
            context,
          ),
          3000,
        );
        expect(reply).toEqual({ stopReason: "end_turn" });
        expect(updates.slice(start)).toContainEqual({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "REPLACEMENT_OK" },
        });
      }
    } finally {
      agent.shutdown();
    }
  });

  test("cancels an execute command before its first turn event", async () => {
    const server = new FakeAppServer();
    const agent = createAgent(server);
    const { context } = createContext();

    try {
      const session = await openSession(agent, context);
      const commandStarted = server.nextCommand();
      const prompt = agent.prompt(
        {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "/init" }],
        },
        context,
      );
      await commandStarted;
      await agent.cancel({ sessionId: session.sessionId });

      expect(await prompt).toEqual({ stopReason: "cancelled" });
    } finally {
      agent.shutdown();
    }
  });
});
