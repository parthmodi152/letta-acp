import {
  type AgentContext,
  type CancelNotification,
  type CloseSessionRequest,
  type CloseSessionResponse,
  type DeleteSessionRequest,
  type DeleteSessionResponse,
  type ForkSessionRequest,
  type ForkSessionResponse,
  type ContentBlock,
  type InitializeRequest,
  type InitializeResponse,
  type LoadSessionRequest,
  type LoadSessionResponse,
  type ListSessionsRequest,
  type ListSessionsResponse,
  type McpServer,
  methods,
  type NewSessionRequest,
  type NewSessionResponse,
  PROTOCOL_VERSION,
  type PromptRequest,
  type PromptResponse,
  type RequestPermissionResponse,
  type SessionConfigOption,
  type SetSessionConfigOptionRequest,
  type SetSessionConfigOptionResponse,
  type SetSessionModeRequest,
  type SetSessionModeResponse,
  type StopReason,
  type ToolCallContent,
} from "@agentclientprotocol/sdk";
import {
  type CanUseToolContext,
  type CanUseToolResponse,
  LettaAgentClient,
  type LettaCodeSession,
  type LettaConversation,
  type McpServers,
  type MessageContentItem,
  type PermissionMode,
  type SDKMessage,
  type SDKQueueItem,
  type SDKResultMessage,
} from "@letta-ai/letta-agent-sdk";
import { authMethodsForClient } from "./auth.js";
import { DEFAULT_ACP_MODEL, type LettaAcpConfig } from "./config.js";
import {
  createEditorTools,
  type EditorFsCapabilities,
} from "./editor-tools.js";
import { historyToUpdates } from "./history-replay.js";
import {
  buildModelCatalog,
  DEFAULT_EFFORT_VALUE,
  entryForChoice,
  isReasoningEffort,
  legacyModelState,
  MODEL_CONFIG_ID,
  modelConfigOptions,
  REASONING_EFFORT_CONFIG_ID,
  resolveSelection,
} from "./model-options.js";
import { MessageInbox } from "./message-inbox.js";
import {
  imageMarkdown,
  readImageFile,
  resultMentionsImage,
  saveToolImage,
  storedToolImages,
  toolImagePath,
  type ToolImage,
} from "./tool-images.js";
import { isPlanTool, PlanTracker } from "./plan.js";
import { withAcpRequestTimeout } from "./request-timeout.js";
import { SessionRegistry } from "./session-registry.js";
import {
  compactionText,
  parseStreamEvent,
  retryText,
  type CompactionSummary,
  type RetryReport,
  type UsageReport,
} from "./stream-events.js";
import {
  editorReadAutoAllows,
  isSessionModeId,
  modeAutoAllows,
  PERMISSION_MODE_CONFIG_ID,
  permissionModeConfigOption,
  sessionModeState,
} from "./session-modes.js";
import {
  buildAvailableCommands,
  isExecuteCommand,
  isTextCommand,
} from "./slash-commands.js";
import {
  accumulateToolInput,
  isTerminalOutputTool,
  parseToolOutput,
  type ToolCallInputState,
  toolDiffContent,
  toolKind,
  toolLocations,
  toolOutputLine,
  toolResultContent,
  toolTitle,
} from "./tool-info.js";

interface StreamedToolCall {
  /** First non-placeholder name seen for this tool call. */
  toolName: string;
  input: ToolCallInputState;
  /** Whether the ACP card already contains a native diff. */
  hasDiff: boolean;
  /** Whether terminal_info has created a native terminal for this call. */
  hasTerminal: boolean;
}

interface AcpSessionState {
  session: LettaCodeSession;
  /** Connection-scoped ACP client context used for requests and notifications. */
  clientContext: AgentContext;
  /** Most recent tool_call streamed, to correlate permission requests. */
  lastToolCall: { id: string; name: string } | null;
  /** Tool calls still streaming or running, keyed by tool call id. */
  toolCalls: Map<string, StreamedToolCall>;
  /** Tools the user chose "always allow" for, scoped to this session. */
  alwaysAllowed: Set<string>;
  cancelled: boolean;
  /** Settles when the session's latest prompt has finished. */
  promptChain: Promise<void>;
  /** The abort sent by the last cancel, until the server has processed it. */
  abortInFlight: Promise<void> | null;
  /** Run ids observed during the current prompt's turn. */
  turnRunIds: Set<string>;
  /**
   * Runs of cancelled turns whose terminal result was never consumed. Their
   * late events and results are dropped instead of being taken for the next
   * prompt's (which would shift every later reply by one).
   */
  staleRunIds: Set<string>;
  /** True only while a session/prompt handler is running. */
  promptActive: boolean;
  /** The prompt is waiting on a turn the harness started for a mod command. */
  awaitingHarnessTurn?: boolean;
  /**
   * A cancel ended the ACP prompt but the Letta turn is still running, for
   * the steer grace period. A prompt that arrives meanwhile is queued into
   * that turn; otherwise the timer aborts it. Tool permissions wait on `hold`
   * (true once a prompt adopts the turn, false once it is aborted).
   */
  softCancel?: {
    timer: ReturnType<typeof setTimeout>;
    hold: Promise<boolean>;
    settle: (adopted: boolean) => void;
  } | null;
  /** Set by a soft cancel to make the running pump hand the turn back. */
  detachRequested?: boolean;
  /**
   * The current prompt's message, identified by the otid it was sent with.
   * The prompt ends on the first result after the runtime dequeued it, which
   * may be the result of a turn that was already running (a steer, or a goal
   * continuation the message was folded into).
   */
  sentMessage?: {
    otid: string;
    /** Seen in a queue_update: it waited behind a running turn. */
    queued: boolean;
    /** Left the queue after being seen in it. */
    dequeued: boolean;
    /** Runs of the turn that was running when the message was sent. */
    priorRunIds: Set<string>;
  } | null;
  /** ACP session mode; enforced adapter-side in the permission callback. */
  modeId: PermissionMode;
  /** Session working directory (for project skill discovery). */
  cwd: string;
  /** Model handle reported by the runtime, updated after ACP model changes. */
  currentModel: string | undefined;
  /** Catalog entry id last applied through updateModel (pins model + effort). */
  currentModelEntryId: string | undefined;
  /** Model settings last reported by updateModel (reasoning_effort, context_window). */
  currentModelSettings: Record<string, unknown> | null | undefined;
  /** The agent's task list, reported to the client as an ACP plan. */
  plan: PlanTracker;
  /** Model cost reported so far in this session, in USD. */
  costUsd: number;
  /** Compactions started in this session, for unique compaction ids. */
  compactions: number;
  /** The compaction whose summary has not arrived yet. */
  openCompactionId: string | null;
  /**
   * Messages read from the SDK while a prompt is running. One reader per
   * session drains the SDK; outside a prompt it forwards them directly.
   */
  inbox: MessageInbox<SDKMessage>;
  /**
   * Tool results that returned an image the stream left out. Their images
   * are read from the stored messages before the next message is shown.
   */
  pendingImages: Array<{ toolCallId: string; content: ToolCallContent[] }>;
  /** Context window the server last reported for this conversation. */
  reportedContextWindow: number | undefined;
  /** Catalog context window, looked up once per model. */
  catalogContextWindow: { model: string; size: number | undefined } | null;
}

/** Max history messages replayed on session/load. */
const LOAD_HISTORY_LIMIT = 200;
/** Number of project sessions returned in one session/list page. */
const SESSION_LIST_PAGE_SIZE = 50;

/**
 * Liveness bound for permission requests raised outside a prompt turn.
 *
 * ACP scopes `session/request_permission` to a turn: the client's only
 * obligation to answer is the `session/cancel` contract, which no longer
 * applies once the prompt handler has returned. A client with no out-of-turn
 * permission UI would leave the Letta-side tool call pending forever, so those
 * requests get a generous-but-finite deadline. In-turn requests stay unbounded.
 */
const DEFAULT_OUT_OF_TURN_PERMISSION_TIMEOUT_MS = 300_000;

/**
 * How long to keep a turn open after the server reported completion while
 * still listing active runs. Long enough to cover the next model step, short
 * enough that a server which never speaks again cannot wedge the prompt.
 */
const DEFAULT_PREMATURE_RESULT_GRACE_MS = 30_000;
/**
 * After a cancel, how long a prompt keeps draining the aborted turn for its
 * terminal result once the server reports idle. Letta Cloud sends that result
 * right after the idle status even when the turn had no visible output yet.
 */
const DEFAULT_CANCEL_DRAIN_MS = 5_000;

/** The harness has no command by that name. */
const UNKNOWN_COMMAND = Symbol("unknown-command");
/** A mod command returned a prompt, and the harness started its turn. */
const MOD_PROMPT_STARTED = Symbol("mod-prompt-started");

/** Safety cap on stream rounds while waiting for a slash command to finish. */
const EXECUTE_COMMAND_MAX_ROUNDS = 50;

type PumpOutcome =
  | { kind: "result"; result: SDKResultMessage }
  /** Turn reported complete while the server still listed active runs. */
  | { kind: "premature"; result: SDKResultMessage }
  | { kind: "idle" }
  | { kind: "stream_end" }
  /** A soft cancel took the turn away from this prompt; it keeps running. */
  | { kind: "detached" };

const IMAGE_MEDIA_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
]);

/**
 * Bridges ACP v1 (agent side) onto a Letta agent via the Letta Agent SDK.
 *
 * One process serves one ACP connection. Every ACP session becomes a new
 * Letta conversation on a single underlying Letta agent, so the agent's
 * memory persists across sessions and editors.
 */
export class LettaAcpAgent {
  private readonly config: LettaAcpConfig;
  private readonly client: LettaAgentClient;
  private readonly outOfTurnPermissionTimeoutMs: number;
  private readonly prematureResultGraceMs: number;
  private readonly cancelDrainMs: number;
  private readonly steerGraceMs: number;
  private readonly sessionRegistry: SessionRegistry;
  private readonly sessions = new Map<string, AcpSessionState>();
  private agentIdPromise: Promise<string> | null = null;
  private supportsTerminalOutput = false;
  private clientFsCaps: EditorFsCapabilities = {
    readTextFile: false,
    writeTextFile: false,
  };
  /** Whether tool images are also posted as markdown links to their file. */
  private imageLinks = false;
  /** Optional client features that change how updates are reported. */
  private clientFeatures = {
    compactionUpdates: false,
    notices: false,
  };

  constructor(config: LettaAcpConfig) {
    this.config = config;
    this.client = new LettaAgentClient(config.clientOptions);
    this.sessionRegistry = new SessionRegistry(
      config.sessionRegistryDir ?? null,
      config.sessionRegistryScope ?? config.clientOptions.backend ?? "local",
    );
    this.outOfTurnPermissionTimeoutMs =
      config.outOfTurnPermissionTimeoutMs ??
      DEFAULT_OUT_OF_TURN_PERMISSION_TIMEOUT_MS;
    this.prematureResultGraceMs =
      config.prematureResultGraceMs ?? DEFAULT_PREMATURE_RESULT_GRACE_MS;
    this.cancelDrainMs = config.cancelDrainMs ?? DEFAULT_CANCEL_DRAIN_MS;
    this.steerGraceMs = config.steerGraceMs ?? 0;
  }

  /**
   * True for an event or result of a cancelled turn whose result was never
   * consumed. A result settles its runs, so they stop being tracked.
   */
  private isStaleRunMessage(state: AcpSessionState, message: SDKMessage): boolean {
    if (state.staleRunIds.size === 0) return false;
    const runIds = messageRunIds(message);
    if (message.type === "loop_status" || runIds.length === 0) return false;
    if (!runIds.every((runId) => state.staleRunIds.has(runId))) return false;
    if (message.type === "result") {
      for (const runId of runIds) state.staleRunIds.delete(runId);
      log(`dropped late result of cancelled run ${runIds.join(", ")}`);
    }
    return true;
  }

  async initialize(params: InitializeRequest): Promise<InitializeResponse> {
    const fs = params.clientCapabilities?.fs;
    this.supportsTerminalOutput =
      params.clientCapabilities?._meta?.terminal_output === true;
    this.clientFsCaps = {
      readTextFile: fs?.readTextFile === true,
      writeTextFile: fs?.writeTextFile === true,
    };
    // `session.compaction` and `session.notices` postdate the pinned ACP SDK
    // types, so read them off the raw capability object.
    const sessionCaps = params.clientCapabilities?.session as
      | Record<string, unknown>
      | null
      | undefined;
    this.imageLinks =
      this.config.imageLinks === "on" ||
      ((this.config.imageLinks ?? "auto") === "auto" &&
        params.clientInfo?.name === "t3-code");
    this.clientFeatures = {
      compactionUpdates: sessionCaps?.compaction != null,
      notices: sessionCaps?.notices != null,
    };
    const requested = params.protocolVersion;
    const protocolVersion =
      typeof requested === "number" && requested < PROTOCOL_VERSION
        ? requested
        : PROTOCOL_VERSION;
    return {
      protocolVersion,
      agentCapabilities: {
        loadSession: true,
        sessionCapabilities: { list: {}, close: {}, delete: {}, fork: {} },
        mcpCapabilities: { http: true, sse: true },
        promptCapabilities: {
          image: true,
          embeddedContext: true,
        },
      },
      authMethods: authMethodsForClient(
        params.clientCapabilities,
        this.config.clientOptions.backend === "local",
      ),
    };
  }

  async newSession(
    params: NewSessionRequest,
    cx: AgentContext,
  ): Promise<NewSessionResponse> {
    const agentId = await this.ensureAgent();
    const { sessionId, state } = await this.openSession({
      cwd: params.cwd,
      resumeId: null,
      agentId,
      clientContext: cx,
      mcpServers: params.mcpServers,
    });
    await this.rememberSession(agentId, sessionId, params.cwd);
    log(`session ${sessionId} -> agent ${agentId} (cwd: ${params.cwd})`);
    this.announceCommands(sessionId, cx);
    const response: NewSessionResponse = {
      sessionId,
      ...(await this.sessionSetup(state)),
    };
    return response;
  }

  async loadSession(
    params: LoadSessionRequest,
    cx: AgentContext,
  ): Promise<LoadSessionResponse> {
    assertConversationId(params.sessionId, "load");
    const { sessionId, state } = await this.openSession({
      cwd: params.cwd,
      resumeId: params.sessionId,
      agentId: null,
      clientContext: cx,
      mcpServers: params.mcpServers,
    });
    await this.rememberSession(state.session.agentId, sessionId, params.cwd);
    const history = await state.session.listMessages({
      order: "desc",
      limit: LOAD_HISTORY_LIMIT,
    });
    if (history.hasMore) {
      log(
        `session ${sessionId} has more than ${LOAD_HISTORY_LIMIT} messages; replaying the most recent ${LOAD_HISTORY_LIMIT}`,
      );
    }
    const updates = historyToUpdates([...history.messages].reverse(), {
      terminalOutput: this.supportsTerminalOutput,
      cwd: state.cwd,
    });
    for (const update of updates) {
      await cx.notify(methods.client.session.update, { sessionId, update });
    }
    log(`loaded session ${sessionId} (${updates.length} replayed updates)`);
    this.announceCommands(sessionId, cx);
    const response: LoadSessionResponse = await this.sessionSetup(state);
    return response;
  }

  async listSessions(params: ListSessionsRequest): Promise<ListSessionsResponse> {
    const agentId = await this.ensureAgent();
    const records = await this.sessionRegistry.list(agentId, params.cwd);
    if (records.length === 0) return { sessions: [] };

    let start = 0;
    if (params.cursor) {
      const previous = records.findIndex((record) => record.sessionId === params.cursor);
      if (previous < 0) throw new Error("Invalid session/list cursor");
      start = previous + 1;
    }

    // Bound each page by registry records rather than successful lookups. Stale
    // or temporarily unavailable conversations must not turn one list request
    // into an unbounded sequence of management calls.
    const page = records.slice(start, start + SESSION_LIST_PAGE_SIZE);
    const resolved = await Promise.all(
      page.map(async (record) => {
        let conversation: LettaConversation;
        try {
          conversation = await this.client.conversations.retrieve(record.sessionId);
        } catch (error) {
          // A deleted conversation is indistinguishable here from a transient
          // management failure. Omit it for this response but retain the local
          // record so an outage cannot erase discoverability permanently.
          log(`cannot resolve listed session ${record.sessionId}: ${String(error)}`);
          return null;
        }
        if (conversation.agent_id !== agentId || conversation.archived === true) {
          await this.sessionRegistry.remove(record.sessionId);
          return null;
        }
        const title = sessionTitle(conversation);
        const updatedAt =
          conversation.last_message_at ??
          conversation.updated_at ??
          conversation.created_at ??
          record.recordedAt;
        return {
          sessionId: record.sessionId,
          cwd: record.cwd,
          ...(title ? { title } : {}),
          ...(updatedAt ? { updatedAt } : {}),
        };
      }),
    );
    const sessions = resolved.filter(
      (session): session is NonNullable<typeof session> => session !== null,
    );
    const last = page.at(-1);

    return {
      sessions,
      ...(last && start + page.length < records.length
        ? { nextCursor: last.sessionId }
        : {}),
    };
  }

  /** Ends the session's work and releases its Letta session. */
  async closeSession(params: CloseSessionRequest): Promise<CloseSessionResponse> {
    const state = this.sessions.get(params.sessionId);
    if (!state) return {};
    if (state.softCancel) await this.abandonDetachedTurn(params.sessionId, state, false);
    else if (state.promptActive) await this.hardCancel(params.sessionId, state);
    this.sessions.delete(params.sessionId);
    state.session.close();
    log(`closed session ${params.sessionId}`);
    return {};
  }

  /**
   * Removes a session from session/list. The Letta conversation is archived
   * rather than destroyed, so its history and the agent's memory stay.
   */
  async deleteSession(params: DeleteSessionRequest): Promise<DeleteSessionResponse> {
    assertConversationId(params.sessionId, "delete");
    await this.closeSession({ sessionId: params.sessionId });
    await this.client.conversations.update(params.sessionId, { archived: true });
    await this.sessionRegistry.remove(params.sessionId);
    log(`archived session ${params.sessionId}`);
    return {};
  }

  /**
   * Copies the session's conversation into a new one, which is ready to
   * prompt. Letta forks the in-context history; the client keeps its own copy
   * of the transcript, so nothing is replayed.
   */
  async forkSession(
    params: ForkSessionRequest,
    cx: AgentContext,
  ): Promise<ForkSessionResponse> {
    assertConversationId(params.sessionId, "fork");
    const fork = await this.client.conversations.fork(params.sessionId);
    const { sessionId, state } = await this.openSession({
      cwd: params.cwd,
      resumeId: fork.id,
      agentId: null,
      clientContext: cx,
      mcpServers: params.mcpServers ?? [],
    });
    await this.rememberSession(state.session.agentId, sessionId, params.cwd);
    log(`forked session ${params.sessionId} -> ${sessionId}`);
    this.announceCommands(sessionId, cx);
    return { sessionId, ...(await this.sessionSetup(state)) };
  }

  private async rememberSession(
    agentId: string | null | undefined,
    sessionId: string,
    cwd: string,
  ): Promise<void> {
    if (!agentId) {
      log(`cannot record session ${sessionId}: runtime did not report an agent id`);
      return;
    }
    try {
      await this.sessionRegistry.record(agentId, sessionId, cwd);
    } catch (error) {
      // Session creation/loading already succeeded. Do not turn a local metadata
      // write failure into a retry that creates a duplicate conversation.
      log(`failed to record session ${sessionId}: ${String(error)}`);
    }
  }

  /** The model catalog and the session's resolved model/effort within it. */
  private async modelView(state: AcpSessionState) {
    const { entries } = await state.session.listModels();
    const catalog = buildModelCatalog(entries);
    const selection = resolveSelection(catalog, {
      entryId: state.currentModelEntryId,
      model: state.currentModel,
      settings: state.currentModelSettings,
    });
    return { catalog, selection };
  }

  private configOptionsFor(
    state: AcpSessionState,
    view: Awaited<ReturnType<LettaAcpAgent["modelView"]>>,
  ): SessionConfigOption[] {
    // A custom model handle may be active even when it is not in the curated
    // catalog; modelConfigOptions keeps it as the current value rather than
    // falsely presenting a different selection.
    return [
      ...(this.config.permissionSelector === false
        ? []
        : [permissionModeConfigOption(state.modeId)]),
      ...modelConfigOptions(view.catalog, view.selection, state.currentModel),
    ];
  }

  private async sessionConfigOptions(
    state: AcpSessionState,
  ): Promise<SessionConfigOption[]> {
    return this.configOptionsFor(state, await this.modelView(state));
  }

  /**
   * The mode, config option, and legacy model state returned when a session is
   * created or loaded. `models` is ACP's legacy model API; it is included only
   * to carry each model's reasoning tiers (see legacyModelState).
   */
  private async sessionSetup(state: AcpSessionState) {
    const view = await this.modelView(state);
    const models = legacyModelState(view.catalog, view.selection);
    return {
      ...(this.config.permissionSelector === false
        ? {}
        : { modes: sessionModeState(state.modeId) }),
      configOptions: this.configOptionsFor(state, view),
      ...(models ? { models } : {}),
    };
  }

  /**
   * Applies a model/effort choice. Either argument may be omitted to keep the
   * session's current value; the pair resolves to one catalog entry id.
   */
  private async applyModelChoice(
    state: AcpSessionState,
    choice: { model?: string; effort?: string },
  ): Promise<void> {
    const { entries } = await state.session.listModels();
    const catalog = buildModelCatalog(entries);
    const current = resolveSelection(catalog, {
      entryId: state.currentModelEntryId,
      model: state.currentModel,
      settings: state.currentModelSettings,
    });
    const groupValue = choice.model ?? current?.group.value;
    if (!groupValue) {
      throw new Error("No current model to apply a reasoning effort to");
    }
    if (choice.effort === DEFAULT_EFFORT_VALUE && choice.model === undefined) {
      // "Default" is the only choice offered for a model without tiers.
      if (current && current.group.tiers.length > 0) {
        throw new Error(`${current.group.label} needs an explicit reasoning effort`);
      }
      return;
    }
    if (choice.effort !== undefined) {
      const group = catalog.groupByEntryId.get(groupValue);
      const offered =
        isReasoningEffort(choice.effort) &&
        group?.tiers.some((tier) => tier.effort === choice.effort);
      if (!offered) {
        throw new Error(
          `${group?.label ?? groupValue} has no ${choice.effort} reasoning effort`,
        );
      }
    }
    // A new model keeps the current effort when it offers that tier.
    const effort = choice.effort ?? current?.effort;
    const entryId = entryForChoice(
      catalog,
      groupValue,
      isReasoningEffort(effort) ? effort : undefined,
    );
    // Not a catalog group: a custom handle passed straight through.
    await this.applyModel(state, entryId ?? groupValue);
  }

  private async applyModel(state: AcpSessionState, model: string): Promise<void> {
    const result = await state.session.updateModel(model);
    state.currentModel = result.modelHandle ?? result.modelId ?? model;
    state.currentModelEntryId = result.modelId ?? model;
    if (result.modelSettings !== undefined) {
      state.currentModelSettings = result.modelSettings;
    }
    // A different model has its own window; wait for the server to report it.
    state.reportedContextWindow = undefined;
  }

  async setSessionConfigOption(
    params: SetSessionConfigOptionRequest,
  ): Promise<SetSessionConfigOptionResponse> {
    const started = Date.now();
    log(
      `session ${params.sessionId} set ${params.configId} = ${String(params.value)}`,
    );
    try {
      return await this.applySessionConfigOption(params);
    } catch (error) {
      log(
        `session ${params.sessionId} set ${params.configId} failed after ${Date.now() - started}ms: ${error instanceof Error ? error.message : String(error)}`,
      );
      throw error;
    }
  }

  private async applySessionConfigOption(
    params: SetSessionConfigOptionRequest,
  ): Promise<SetSessionConfigOptionResponse> {
    const state = this.sessions.get(params.sessionId);
    if (!state) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }
    if (params.configId === PERMISSION_MODE_CONFIG_ID) {
      if (typeof params.value !== "string" || !isSessionModeId(params.value)) {
        throw new Error(`Unknown permission mode: ${String(params.value)}`);
      }
      state.modeId = params.value;
      log(`session ${params.sessionId} mode -> ${params.value}`);
      return { configOptions: await this.sessionConfigOptions(state) };
    }
    if (params.configId === MODEL_CONFIG_ID) {
      if (typeof params.value !== "string") {
        throw new Error("The model configuration option requires a model id");
      }
      await this.applyModelChoice(state, { model: params.value });
      log(
        `session ${params.sessionId} model -> ${state.currentModelEntryId ?? state.currentModel}`,
      );
      return { configOptions: await this.sessionConfigOptions(state) };
    }
    if (params.configId === REASONING_EFFORT_CONFIG_ID) {
      if (typeof params.value !== "string") {
        throw new Error("The reasoning effort option requires an effort value");
      }
      await this.applyModelChoice(state, { effort: params.value });
      log(
        `session ${params.sessionId} effort -> ${params.value} (${state.currentModelEntryId})`,
      );
      return { configOptions: await this.sessionConfigOptions(state) };
    }
    throw new Error(`Unknown session configuration option: ${params.configId}`);
  }

  async setSessionMode(
    params: SetSessionModeRequest,
    cx: AgentContext,
  ): Promise<SetSessionModeResponse> {
    const state = this.sessions.get(params.sessionId);
    if (!state) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }
    if (!isSessionModeId(params.modeId)) {
      throw new Error(`Unknown mode: ${params.modeId}`);
    }
    state.modeId = params.modeId;
    log(`session ${params.sessionId} mode -> ${params.modeId}`);
    void cx.notify(methods.client.session.update, {
      sessionId: params.sessionId,
      update: {
        sessionUpdate: "current_mode_update",
        currentModeId: params.modeId,
      },
    });
    return {};
  }

  /**
   * Creates or resumes the underlying Letta session and registers state.
   *
   * The ACP session id is the Letta conversation id, which is what makes
   * session/load work across adapter restarts with no local persistence. The
   * id is only known after the runtime initializes, so callbacks close over a
   * mutable ref that is filled in before any of them can fire (they only run
   * during prompts).
   *
   * The harness always runs with permissionMode "standard"; the ACP-selected
   * mode is enforced adapter-side (see session-modes.ts).
   */
  private async openSession(options: {
    cwd: string;
    resumeId: string | null;
    agentId: string | null;
    clientContext: AgentContext;
    mcpServers?: readonly McpServer[];
  }): Promise<{ sessionId: string; state: AcpSessionState }> {
    const ref = { sessionId: options.resumeId ?? "" };
    const editorTools = createEditorTools(this.clientFsCaps, {
      getSessionId: () => ref.sessionId,
      getClientContext: () =>
        this.sessions.get(ref.sessionId)?.clientContext ?? null,
    });
    const mcpServers = toSdkMcpServers(options.mcpServers);
    const sessionOptions = {
      cwd: options.cwd,
      model: this.config.model,
      permissionMode: "standard" as const,
      canUseTool: (
        toolName: string,
        toolInput: Record<string, unknown>,
        context?: CanUseToolContext,
      ) =>
        this.requestToolPermission(
          ref.sessionId,
          toolName,
          toolInput,
          context,
        ),
      ...(editorTools.length > 0 ? { tools: editorTools } : {}),
      ...(Object.keys(mcpServers).length > 0 ? { mcpServers } : {}),
    };
    const session = options.resumeId
      ? this.client.resumeSession(options.resumeId, sessionOptions)
      : this.client.createSession(options.agentId ?? "", sessionOptions);
    // Force runtime initialization so the conversation id and active model exist.
    const bootstrap = await session.bootstrapState({ limit: 1 });
    const sessionId = options.resumeId ?? session.conversationId;
    if (!sessionId) {
      session.close();
      throw new Error("Letta session did not report a conversation id");
    }
    ref.sessionId = sessionId;
    if (editorTools.length > 0) {
      log(
        `editor fs tools enabled: ${editorTools.map((tool) => tool.name).join(", ")}`,
      );
    }
    const state: AcpSessionState = {
      session,
      clientContext: options.clientContext,
      lastToolCall: null,
      toolCalls: new Map(),
      alwaysAllowed: new Set(),
      cancelled: false,
      promptChain: Promise.resolve(),
      abortInFlight: null,
      turnRunIds: new Set(),
      staleRunIds: new Set(),
      promptActive: false,
      modeId: this.config.permissionMode,
      cwd: options.cwd,
      currentModel: bootstrap.model ?? this.config.model,
      currentModelEntryId: undefined,
      currentModelSettings: runtimeModelSettings(session),
      plan: new PlanTracker(),
      costUsd: 0,
      compactions: 0,
      openCompactionId: null,
      catalogContextWindow: null,
      reportedContextWindow: undefined,
      inbox: new MessageInbox(),
      pendingImages: [],
    };
    // A client may load the same conversation again while its prior ACP
    // session is still open. Close the displaced SDK session before replacing
    // it so its sockets and per-session MCP subprocesses are not orphaned.
    const displaced = this.sessions.get(sessionId);
    if (displaced && displaced.session !== session) {
      displaced.session.close();
    }
    this.sessions.set(sessionId, state);
    this.startReader(sessionId, state);
    return { sessionId, state };
  }

  private announceCommands(sessionId: string, cx: AgentContext): void {
    const cwd = this.sessions.get(sessionId)?.cwd ?? process.cwd();
    const availableCommands = buildAvailableCommands(cwd);
    log(`advertising ${availableCommands.length} slash commands`);
    void cx.notify(methods.client.session.update, {
      sessionId,
      update: {
        sessionUpdate: "available_commands_update",
        availableCommands,
      },
    });
  }

  async prompt(
    params: PromptRequest,
    cx: AgentContext,
  ): Promise<PromptResponse> {
    const state = this.sessions.get(params.sessionId);
    if (!state) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }
    // A prompt arrived within the steer grace period: it decides what happens
    // to the detached turn, so the abort timer must not fire meanwhile.
    if (state.softCancel) clearTimeout(state.softCancel.timer);
    // One prompt at a time per session: a prompt that arrives while a
    // cancelled one is still draining its turn waits for it, so the two can
    // never read each other's events from the shared SDK stream.
    const previous = state.promptChain;
    let release!: () => void;
    state.promptChain = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await this.runPrompt(params, state, cx);
    } finally {
      release();
    }
  }

  private async runPrompt(
    params: PromptRequest,
    state: AcpSessionState,
    cx: AgentContext,
  ): Promise<PromptResponse> {
    state.clientContext = cx;
    // A prompt right after a soft cancel is a steer: queue it into the turn
    // that is still running. Commands and attachments cannot join a running
    // turn, so for those the turn is aborted first, as before.
    let steering = false;
    if (state.softCancel) {
      if (isSteerablePrompt(params)) {
        this.adoptDetachedTurn(state);
        steering = true;
      } else {
        await this.abandonDetachedTurn(params.sessionId, state, true);
      }
    }
    // A steer (cancel, then an immediate new prompt) must not send before the
    // server has processed the abort: aborting clears the server's input
    // queue, which would silently drop the new message.
    if (state.abortInFlight) await state.abortInFlight;
    state.cancelled = false;
    state.promptActive = true;
    state.sentMessage = null;
    let detached = false;
    // A steered prompt continues the running turn, whose runs stay this
    // prompt's.
    if (!steering) state.turnRunIds = new Set();

    try {
      const commandResponse = steering
        ? null
        : await this.maybeRunCommand(params, state, cx);
      if (commandResponse && commandResponse !== MOD_PROMPT_STARTED) {
        return commandResponse;
      }
      // A mod command that returns a prompt has already started its turn.
      state.awaitingHarnessTurn = commandResponse === MOD_PROMPT_STARTED;
      if (!commandResponse) {
        const otid = crypto.randomUUID();
        state.sentMessage = {
          otid,
          queued: false,
          dequeued: false,
          priorRunIds: new Set(state.turnRunIds),
        };
        await state.session.send(toLettaContent(params.prompt), { otid });
        if (steering) {
          log(`steer: queued the new prompt into the running turn`);
          await this.reannounceToolCalls(params.sessionId, state, cx);
        }
      }
      // The Letta app-server transport completes a turn with a recoverable
      // "approval_conflict" result whenever a tool needs user approval. The
      // approval itself resolves concurrently: the server sends a
      // can_use_tool control request, our canUseTool callback forwards it to
      // the ACP client, and once answered the run resumes and its events keep
      // streaming — but with no second terminal result message. So: pump the
      // first round to its result, then pump recovery rounds (blocking on the
      // stream while approvals resolve) until the run goes idle.
      let mode: "turn" | "recovery" = "turn";
      let graceMs: number | undefined;
      while (true) {
        const outcome = await this.pumpStream(
          params.sessionId,
          state,
          cx,
          mode,
          graceMs,
        );
        if (outcome.kind === "detached") {
          detached = true;
          return { stopReason: "cancelled" };
        }
        if (state.cancelled) return { stopReason: "cancelled" };
        switch (outcome.kind) {
          case "premature": {
            // The SDK ends its stream on any result, so the only way to stay
            // with the run is another round — the same recovery the
            // approval_conflict path uses, but on a deadline in case the
            // server really was done.
            const runIds = outcome.result.runIds?.join(", ") ?? "unknown";
            log(`turn reported complete while runs are active (${runIds})`);
            mode = "recovery";
            graceMs = this.prematureResultGraceMs;
            continue;
          }
          case "result": {
            const result = outcome.result;
            if (!result.success && result.errorCode === "approval_conflict") {
              mode = "recovery";
              graceMs = undefined;
              continue;
            }
            return this.toPromptResponse(state, result);
          }
          case "idle":
          case "stream_end":
            return { stopReason: "end_turn" };
        }
      }
    } catch (error) {
      if (state.cancelled) return { stopReason: "cancelled" };
      throw error;
    } finally {
      await this.flushPendingImages(params.sessionId, state, cx);
      await this.settleCompaction(params.sessionId, state, cx).catch((error: unknown) => {
        log(`could not close the compaction card: ${String(error)}`);
      });
      // The client context stays for the lifetime of the session; only the
      // in-a-turn marker is cleared, so late tool traffic still reaches the
      // client but waits on a deadline.
      state.promptActive = false;
      // A detached turn is still running: its messages stay in the inbox for
      // the prompt that steers it, or for the abort's drain.
      if (!detached) {
        state.sentMessage = null;
        // Whatever arrived after the turn's result belongs to no prompt now.
        for (const message of state.inbox.drain()) {
          await this.forwardOutOfTurn(params.sessionId, state, message);
        }
      }
    }
  }

  /** Takes over a soft-cancelled turn for a steering prompt. */
  private adoptDetachedTurn(state: AcpSessionState): void {
    const soft = state.softCancel;
    if (!soft) return;
    clearTimeout(soft.timer);
    state.softCancel = null;
    state.detachRequested = false;
    soft.settle(true);
  }

  /**
   * Aborts a soft-cancelled turn: no steer followed the cancel, or the next
   * prompt cannot be queued into it. Tool calls held meanwhile are denied,
   * and the aborted turn is drained so its result is not taken for the next
   * prompt's. `locked` means the caller already holds the prompt chain.
   */
  private async abandonDetachedTurn(
    sessionId: string,
    state: AcpSessionState,
    locked: boolean,
  ): Promise<void> {
    const soft = state.softCancel;
    if (!soft) return;
    clearTimeout(soft.timer);
    state.softCancel = null;
    state.detachRequested = false;
    soft.settle(false);
    const run = async () => {
      log(`no steer followed the cancel of ${sessionId}; aborting the turn`);
      // Keep the reader feeding the inbox while the abort drains.
      state.promptActive = true;
      try {
        await this.hardCancel(sessionId, state);
        await this.pumpStream(sessionId, state, state.clientContext, "turn");
      } catch (error) {
        log(`draining the aborted turn of ${sessionId} failed: ${String(error)}`);
      } finally {
        state.promptActive = false;
        state.sentMessage = null;
        for (const message of state.inbox.drain()) {
          await this.forwardOutOfTurn(sessionId, state, message);
        }
      }
    };
    if (locked) return run();
    const previous = state.promptChain;
    let release!: () => void;
    state.promptChain = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      await run();
    } finally {
      release();
    }
  }

  /**
   * A steering prompt is a new ACP turn, and the client closed the tool
   * cards of the cancelled one. Tool calls still running are announced again
   * so their results have a card to land on.
   */
  private async reannounceToolCalls(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
  ): Promise<void> {
    for (const [toolCallId, call] of state.toolCalls) {
      await cx.notify(methods.client.session.update, {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId,
          title: toolTitle(call.toolName, call.input.input),
          kind: toolKind(call.toolName),
          status: "in_progress",
        },
      });
    }
  }

  /** Records whether the prompt's message is waiting in the runtime's queue. */
  private trackSentMessage(state: AcpSessionState, queue: readonly SDKQueueItem[]): void {
    const sent = state.sentMessage;
    if (!sent) return;
    if (queue.some((item) => item.clientMessageId === sent.otid)) sent.queued = true;
    else if (sent.queued) sent.dequeued = true;
  }

  /**
   * Handles slash commands that have a non-LLM implementation: /model runs in
   * the adapter, and execute_command-backed commands run in the harness.
   * Returns null for everything else — unknown /names fall through as prompt
   * text, which is how skill invocations reach the model.
   */
  private async maybeRunCommand(
    params: PromptRequest,
    state: AcpSessionState,
    cx: AgentContext,
  ): Promise<PromptResponse | typeof MOD_PROMPT_STARTED | null> {
    const first = params.prompt[0];
    if (params.prompt.length !== 1 || first?.type !== "text") return null;
    const match = first.text.trim().match(/^\/([a-z0-9-]+)(?:\s+([\s\S]*))?$/i);
    if (!match) return null;
    const command = (match[1] ?? "").toLowerCase();
    const argument = match[2]?.trim();

    // Harness commands, plus any other /name that is not a skill or text
    // command: it may be one a mod registered, which only the harness can
    // run. When the harness knows no such command, the text goes to the model.
    if (
      isExecuteCommand(command) ||
      (command !== "model" && !isTextCommand(command, state.cwd))
    ) {
      const response = await this.runExecuteCommand(
        params.sessionId,
        state,
        cx,
        command,
        argument,
      );
      return response === UNKNOWN_COMMAND ? null : response;
    }
    if (command !== "model") return null;

    const reply = async (text: string) => {
      await cx.notify(methods.client.session.update, {
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text },
        },
      });
    };
    try {
      if (!argument) {
        const result = await state.session.listModels();
        const lines = result.entries.map((model) => {
          const marks = [
            model.isDefault ? "default" : null,
            model.free ? "free" : null,
          ]
            .filter(Boolean)
            .join(", ");
          return `- \`${model.handle}\` — ${model.label}${marks ? ` (${marks})` : ""}`;
        });
        await reply(
          `Available models:\n${lines.join("\n")}\n\nSwitch with \`/model <handle>\`.`,
        );
      } else {
        await this.applyModel(state, argument);
        await reply(`Model switched to \`${state.currentModel}\`.`);
        // Keep the client's model and effort selectors in sync with /model.
        await cx.notify(methods.client.session.update, {
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "config_option_update",
            configOptions: await this.sessionConfigOptions(state),
          },
        });
      }
    } catch (error) {
      await reply(
        `/model failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    return { stopReason: "end_turn" };
  }

  /**
   * Dispatches a slash command to the harness via the app-server protocol's
   * execute_command, forwarding any resulting agent-turn events (e.g. /init
   * and /remember run full turns) and finishing on the slash_command_end
   * stream delta that carries the command's output.
   */
  private async runExecuteCommand(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
    command: string,
    argument: string | undefined,
  ): Promise<PromptResponse | typeof UNKNOWN_COMMAND | typeof MOD_PROMPT_STARTED> {
    log(`executing /${command} via execute_command`);
    await state.session.sendCommand({
      type: "execute_command",
      command_id: command,
      request_id: `acp-${crypto.randomUUID()}`,
      runtime: {
        agent_id: state.session.agentId,
        conversation_id: state.session.conversationId,
      },
      ...(argument ? { args: argument } : {}),
    });

    let nestedTurns = 0;
    while (true) {
      if (state.cancelled) return { stopReason: "cancelled" };
      const step = await state.inbox.next();
      if (!step || step.done) break;
      const message = step.value;
      if (state.cancelled) return { stopReason: "cancelled" };
      if (TRACE) log(`trace ${sessionId} command ${describeSdkMessage(message)}`);
      if (message.type === "stream_event") {
        const event = message.event as Record<string, unknown>;
        if (
          event.message_type === "slash_command_end" &&
          event.command_id === command
        ) {
          const output = typeof event.output === "string" ? event.output : "";
          const success = event.success !== false;
          if (!success && output === `Unknown command: ${command}`) {
            log(`/${command} is not a harness or mod command; sending it as text`);
            return UNKNOWN_COMMAND;
          }
          // A mod command that returns a prompt ends here, and the harness
          // then runs that prompt as an ordinary turn.
          if (success && output === `Running /${command}...`) {
            log(`/${command} started a turn`);
            return MOD_PROMPT_STARTED;
          }
          if (output) {
            await cx.notify(methods.client.session.update, {
              sessionId,
              update: {
                sessionUpdate: "agent_message_chunk",
                content: {
                  type: "text",
                  text: success ? output : `/${command} failed: ${output}`,
                },
              },
            });
          }
          return { stopReason: "end_turn" };
        }
        // /compact reports its summary as ordinary stream events.
        await this.forwardStreamEvent(sessionId, state, message.event, cx);
        continue;
      }
      if (message.type === "result") {
        // A background turn (task notification, cron) is not the command's.
        if (message.origin !== undefined) {
          log(`background ${message.origin} turn ended during /${command}`);
          continue;
        }
        // Commands like /init and /remember run a full agent turn. A
        // successful nested result, or a recoverable approval conflict, is
        // not the end of the command: its continuation and slash_command_end
        // follow.
        if (message.success || message.errorCode === "approval_conflict") {
          if (++nestedTurns >= EXECUTE_COMMAND_MAX_ROUNDS) break;
          continue;
        }
        // Do not turn genuine nested-turn failures into 50 opaque recovery
        // rounds followed by "/<command> did not complete".
        return this.toPromptResponse(state, message);
      }
      await this.forwardMessage(sessionId, state, message, cx);
    }
    throw new Error(`/${command} did not complete`);
  }

  /**
   * The session's only SDK reader. The SDK ends each stream() at the result
   * of a turn this session sent, so the reader starts the next one at once.
   * Messages go to the running prompt, or straight to the client when no
   * prompt is running (background turns such as subagent notifications).
   */
  private startReader(sessionId: string, state: AcpSessionState): void {
    void (async () => {
      try {
        while (this.sessions.get(sessionId) === state) {
          let received = false;
          for await (const message of state.session.stream()) {
            received = true;
            if (state.promptActive || state.softCancel) state.inbox.push(message);
            else await this.forwardOutOfTurn(sessionId, state, message);
          }
          // An empty stream means the SDK session closed.
          if (!received) break;
        }
      } catch (error) {
        log(`session ${sessionId} reader stopped: ${String(error)}`);
      } finally {
        state.inbox.close();
      }
    })();
  }

  /** Forwards a message that arrived while no prompt was running. */
  private async forwardOutOfTurn(
    sessionId: string,
    state: AcpSessionState,
    message: SDKMessage,
  ): Promise<void> {
    try {
      if (this.isStaleRunMessage(state, message)) return;
      // Leftovers of a cancelled turn: the client was already told it ended.
      if (state.cancelled) return;
      if (TRACE) log(`trace ${sessionId} idle ${describeSdkMessage(message)}`);
      if (message.type === "result") {
        log(
          `out-of-turn ${message.origin ?? "session"} turn ended (${message.success ? "ok" : message.errorCode ?? "failed"})`,
        );
        await this.flushPendingImages(sessionId, state, state.clientContext);
        await this.settleCompaction(sessionId, state, state.clientContext);
        return;
      }
      await this.forwardMessage(sessionId, state, message, state.clientContext);
    } catch (error) {
      log(`could not forward an out-of-turn message: ${String(error)}`);
    }
  }

  /**
   * Iterates one session.stream() round, forwarding events to the client.
   * In "recovery" mode (post-approval continuation) there is no terminal
   * result message, so loop_status transitions decide when the turn is over.
   * Blocking on the stream while an approval is pending is fine — the
   * canUseTool round-trip resolves concurrently over the control channel.
   */
  private async pumpStream(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
    mode: "turn" | "recovery",
    firstMessageDeadlineMs?: number,
  ): Promise<PumpOutcome> {
    let sawActivity = false;
    let idleStatusCount = 0;
    let activeRunIds: readonly string[] = [];

    let awaitingFirstMessage = true;
    // Set once a cancelled turn reports idle: how long to keep waiting for its
    // terminal result before giving up on it.
    let cancelDrainUntil: number | undefined;
    while (true) {
      // A soft cancel hands the running turn back; its messages stay queued.
      if (state.detachRequested && !state.cancelled) return { kind: "detached" };
      // Only the first message of a grace round is raced: a server that went
      // quiet after contradicting itself must not wedge the prompt.
      const deadlineMs =
        cancelDrainUntil !== undefined
          ? Math.max(0, cancelDrainUntil - Date.now())
          : awaitingFirstMessage
            ? firstMessageDeadlineMs
            : undefined;
      const step = await state.inbox.next(deadlineMs);
      if (!step && state.detachRequested && !state.cancelled) return { kind: "detached" };
      if (!step) {
        if (cancelDrainUntil !== undefined) {
          // The aborted turn's result may still arrive; make sure it cannot
          // be taken for the next prompt's.
          for (const runId of state.turnRunIds) state.staleRunIds.add(runId);
          log(
            `cancelled turn left no result within ${this.cancelDrainMs}ms; ignoring runs ${[...state.turnRunIds].join(", ") || "(none)"}`,
          );
        } else {
          log("no further activity during the grace round; ending turn");
        }
        return { kind: "idle" };
      }
      awaitingFirstMessage = false;
      if (step.done) break;
      const message = step.value;
      if (TRACE) {
        log(`trace ${sessionId} ${mode}${state.cancelled ? " cancelled" : ""} ${describeSdkMessage(message)}`);
      }
      if (this.isStaleRunMessage(state, message)) continue;
      if (message.type === "queue_update") this.trackSentMessage(state, message.queue);
      const sent = state.cancelled ? null : state.sentMessage;
      if (message.type === "result" && sent) {
        // The turn that was running when the prompt's message was sent ended
        // before taking the message in: the message starts a turn of its own,
        // and that turn's result is the prompt's.
        const priorTurn = sent.queued
          ? !sent.dequeued
          : (message.runIds ?? []).some((runId) => sent.priorRunIds.has(runId));
        if (priorTurn) {
          log("the running turn ended before reading the new message; waiting for its turn");
          for (const runId of messageRunIds(message)) sent.priorRunIds.add(runId);
          continue;
        }
      }
      if (message.type === "result" && message.origin !== undefined && sent?.dequeued) {
        // The message was folded into a turn the session did not start (a
        // goal continuation, a task notification): that turn answered it.
        log(`the prompt's message was answered inside a ${message.origin} turn`);
      } else if (
        message.type === "result" &&
        message.origin !== undefined &&
        state.awaitingHarnessTurn
      ) {
        // The turn a mod command started is one the session did not send,
        // so the SDK marks it as background; it is this prompt's turn.
        state.awaitingHarnessTurn = false;
      } else if (message.type === "result" && message.origin !== undefined) {
        // A turn the session did not send (a task notification, a cron
        // prompt) ended. Its events were shown in this prompt; the prompt's
        // own turn is still running, so keep going.
        log(`background ${message.origin} turn ended during a prompt`);
        continue;
      }
      for (const runId of messageRunIds(message)) state.turnRunIds.add(runId);

      if (state.cancelled) {
        // Drain the aborted turn without forwarding it: ACP already told the
        // client the turn was cancelled. Its terminal result ends the drain.
        if (message.type === "result") return { kind: "result", result: message };
        if (message.type === "loop_status" && message.status === "WAITING_ON_INPUT") {
          cancelDrainUntil ??= Date.now() + this.cancelDrainMs;
        }
        continue;
      }

      if (message.type === "result") {
        // An idle loop status that still lists active runs makes the SDK
        // synthesize a successful, stop-reason-less turn result. Taking it at
        // face value ends the prompt mid-run, stranding the tool calls that
        // are still streaming and pushing their approvals outside the turn.
        if (
          message.success &&
          message.stopReason == null &&
          activeRunIds.length > 0
        ) {
          return { kind: "premature", result: message };
        }
        return { kind: "result", result: message };
      }
      if (message.type === "loop_status") {
        activeRunIds = message.activeRunIds;
        if (mode === "recovery") {
          if (message.status === "WAITING_ON_INPUT") {
            idleStatusCount += 1;
            // The first idle status can be stale (queued before the resume);
            // trust it once we've seen real activity or it repeats.
            if (sawActivity || idleStatusCount >= 2) {
              return { kind: "idle" };
            }
            // A grace round is the contradicted-completion path. An empty idle
            // there is the server confirming the listed runs will not continue.
            // SDK 0.8 settles those run ids with the premature result, so the
            // continuation is dropped and a second idle never arrives.
            if (
              firstMessageDeadlineMs !== undefined &&
              message.activeRunIds.length === 0
            ) {
              return { kind: "idle" };
            }
          }
          continue;
        }
      }
      const forwarded = await this.forwardMessage(
        sessionId,
        state,
        message,
        cx,
      );
      sawActivity = sawActivity || forwarded;
    }
    return { kind: "stream_end" };
  }

  async cancel(params: CancelNotification): Promise<void> {
    const state = this.sessions.get(params.sessionId);
    if (!state) return;
    // Already detached: the grace timer decides.
    if (state.softCancel) return;
    // A cancel may be the first half of a steer. End the ACP prompt now, so
    // the client is never kept waiting on the abort, but leave the Letta turn
    // running briefly for a prompt to join it.
    if (
      this.steerGraceMs > 0 &&
      state.promptActive &&
      !state.cancelled &&
      state.sentMessage &&
      !state.awaitingHarnessTurn
    ) {
      let settle!: (adopted: boolean) => void;
      const hold = new Promise<boolean>((resolve) => {
        settle = resolve;
      });
      const timer = setTimeout(() => {
        void this.abandonDetachedTurn(params.sessionId, state, false);
      }, this.steerGraceMs);
      state.softCancel = { timer, hold, settle };
      state.detachRequested = true;
      state.inbox.wake();
      log(`cancel of ${params.sessionId}: keeping the turn ${this.steerGraceMs}ms for a steer`);
      return;
    }
    await this.hardCancel(params.sessionId, state);
  }

  /** Aborts the running turn on the server. */
  private async hardCancel(sessionId: string, state: AcpSessionState): Promise<void> {
    const params = { sessionId };
    state.cancelled = true;
    // Interrupted tool calls never produce a tool_result, so drop their
    // accumulated arguments here instead of leaking them for the session.
    state.toolCalls.clear();
    const abort = state.session.abort().catch((error: unknown) => {
      log(`abort failed for ${params.sessionId}: ${String(error)}`);
    });
    const tracked = abort.finally(() => {
      if (state.abortInFlight === tracked) state.abortInFlight = null;
    });
    state.abortInFlight = tracked;
    await tracked;
  }

  shutdown(): void {
    for (const state of this.sessions.values()) {
      try {
        state.session.close();
      } catch {
        // best-effort cleanup on connection close
      }
    }
    this.sessions.clear();
  }

  /**
   * Streamed Letta SDK message -> ACP session/update notification.
   * Returns true when the message was substantive turn activity.
   */
  private async forwardMessage(
    sessionId: string,
    state: AcpSessionState,
    message: SDKMessage,
    cx: AgentContext,
  ): Promise<boolean> {
    if (
      state.pendingImages.length > 0 &&
      (message.type === "assistant" ||
        message.type === "reasoning" ||
        message.type === "tool_call")
    ) {
      // The server has stored the tool return by the time the model speaks
      // again, so its image can be shown before what comes next.
      await this.flushPendingImages(sessionId, state, cx);
    }
    if (
      state.openCompactionId &&
      (message.type === "assistant" ||
        message.type === "reasoning" ||
        message.type === "tool_call")
    ) {
      // Letta Cloud announces a compaction but does not stream its summary;
      // the model speaking again means it is done.
      await this.settleCompaction(sessionId, state, cx);
    }
    switch (message.type) {
      case "init":
        log(`turn started (agent ${message.agentId}, model ${message.model})`);
        return false;
      case "assistant":
        await cx.notify(methods.client.session.update, {
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: message.content },
          },
        });
        return true;
      case "reasoning":
        await cx.notify(methods.client.session.update, {
          sessionId,
          update: {
            sessionUpdate: "agent_thought_chunk",
            content: { type: "text", text: message.content },
          },
        });
        return true;
      case "tool_call": {
        const existing = state.toolCalls.get(message.toolCallId);
        // Continuation deltas carry an argument fragment and often no name, so
        // the SDK reports "?" for them: the first real name wins.
        const toolName =
          knownToolName(message.toolName) ??
          existing?.toolName ??
          message.toolName;
        const input = accumulateToolInput(
          existing?.input,
          message.rawArguments,
          message.toolInput,
        );
        const diffContent = toolDiffContent(toolName, input.input);
        const hasDiff = existing?.hasDiff === true || diffContent.length > 0;
        const nativeTerminal =
          this.supportsTerminalOutput && isTerminalOutputTool(toolName);
        const addTerminal = nativeTerminal && existing?.hasTerminal !== true;
        state.toolCalls.set(message.toolCallId, {
          toolName,
          input,
          hasDiff,
          hasTerminal: existing?.hasTerminal === true || addTerminal,
        });
        state.lastToolCall = { id: message.toolCallId, name: toolName };
        // Partial JSON has no usable title or locations, so hold the card
        // steady until the arguments parse rather than flickering through
        // every fragment.
        const worthSending =
          !existing ||
          (input.complete &&
            (!existing.input.complete ||
              existing.input.rawArguments !== input.rawArguments));
        if (!worthSending) return true;
        await cx.notify(methods.client.session.update, {
          sessionId,
          update: {
            sessionUpdate: existing ? "tool_call_update" : "tool_call",
            toolCallId: message.toolCallId,
            title: toolTitle(toolName, input.input),
            kind: toolKind(toolName),
            status: "in_progress",
            rawInput: input.input,
            locations: toolLocations(input.input),
            ...(diffContent.length > 0
              ? { content: diffContent }
              : addTerminal
                ? {
                    content: [{ type: "terminal" as const, terminalId: message.toolCallId }],
                    _meta: {
                      terminal_info: {
                        terminal_id: message.toolCallId,
                        cwd: state.cwd,
                      },
                    },
                  }
                : {}),
          },
        });
        return true;
      }
      case "tool_result": {
        const toolCall = state.toolCalls.get(message.toolCallId);
        const rawOutput = parseToolOutput(message.content);
        state.toolCalls.delete(message.toolCallId);
        const locations = toolCall
          ? toolLocations(toolCall.input.input, toolOutputLine(rawOutput))
          : [];
        const nativeTerminal = toolCall?.hasTerminal === true;
        if (nativeTerminal) {
          // Zed creates the display-only terminal from terminal_info on the
          // initial call, then consumes output and exit as ordered updates.
          await cx.notify(methods.client.session.update, {
            sessionId,
            update: {
              sessionUpdate: "tool_call_update",
              toolCallId: message.toolCallId,
              _meta: {
                terminal_output: {
                  terminal_id: message.toolCallId,
                  data: message.content,
                },
              },
            },
          });
        }
        if (toolCall && !message.isError && isPlanTool(toolCall.toolName)) {
          const changed = state.plan.apply(
            toolCall.toolName,
            toolCall.input.input,
            rawOutput,
          );
          if (changed) {
            await cx.notify(methods.client.session.update, {
              sessionId,
              update: { sessionUpdate: "plan", entries: state.plan.entries() },
            });
          }
        }
        const resultContent = toolResultContent(
          message.toolCallId,
          message.content,
          {
            includeTerminal: nativeTerminal,
            hasDiff: toolCall?.hasDiff,
            isError: message.isError,
          },
        );
        await cx.notify(methods.client.session.update, {
          sessionId,
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId: message.toolCallId,
            status: message.isError ? "failed" : "completed",
            rawOutput,
            ...(locations.length > 0 ? { locations } : {}),
            ...(resultContent.length > 0 ? { content: resultContent } : {}),
            ...(nativeTerminal
              ? {
                  _meta: {
                    terminal_exit: {
                      terminal_id: message.toolCallId,
                      exit_code: message.isError ? 1 : 0,
                      signal: null,
                    },
                  },
                }
              : {}),
          },
        });
        if (toolCall && !message.isError) {
          await this.collectToolImages(
            sessionId,
            state,
            cx,
            message.toolCallId,
            toolCall,
            message.content,
            resultContent,
          );
        }
        return true;
      }
      case "error":
        log(`stream error: ${message.message}`);
        return false;
      case "retry":
        await this.reportRetry(sessionId, cx, {
          attempt: message.attempt,
          maxAttempts: message.maxAttempts,
          delayMs: message.delayMs,
          reason: message.reason,
        });
        return false;
      case "stream_event":
        await this.forwardStreamEvent(sessionId, state, message.event, cx);
        return false;
      default:
        // queue_update, loop_status — no ACP equivalent.
        return false;
    }
  }

  /** Usage, compaction and retry messages the SDK passes through raw. */
  private async forwardStreamEvent(
    sessionId: string,
    state: AcpSessionState,
    event: unknown,
    cx: AgentContext,
  ): Promise<void> {
    const parsed = parseStreamEvent(event);
    if (!parsed) return;
    switch (parsed.kind) {
      case "usage":
        await this.reportUsage(sessionId, state, parsed.usage, cx);
        return;
      case "retry":
        await this.reportRetry(sessionId, cx, parsed.retry);
        return;
      case "compaction_start":
        if (parsed.contextWindow) state.reportedContextWindow = parsed.contextWindow;
        await this.startCompaction(sessionId, state, cx);
        return;
      case "compaction_summary":
        await this.finishCompaction(sessionId, state, parsed, cx);
        return;
    }
  }

  private async reportUsage(
    sessionId: string,
    state: AcpSessionState,
    usage: UsageReport,
    cx: AgentContext,
  ): Promise<void> {
    if (usage.costUsd !== undefined) state.costUsd += usage.costUsd;
    const size =
      state.reportedContextWindow ??
      contextWindowSize(state) ??
      (await this.catalogContextWindow(state));
    await cx.notify(methods.client.session.update, {
      sessionId,
      update: {
        sessionUpdate: "usage_update",
        used: Math.round(usage.contextTokens),
        size: size ?? 0,
        ...(state.costUsd > 0
          ? { cost: { amount: state.costUsd, currency: "USD" } }
          : {}),
      },
    });
  }

  /**
   * The runtime reports the context window only after a model change, so a
   * session on its starting model takes it from the model catalog.
   */
  private async catalogContextWindow(
    state: AcpSessionState,
  ): Promise<number | undefined> {
    const model = `${state.currentModelEntryId ?? ""}\u0000${state.currentModel ?? ""}`;
    if (state.catalogContextWindow?.model === model) {
      return state.catalogContextWindow.size;
    }
    let size: number | undefined;
    try {
      size = (await this.modelView(state)).selection?.group.contextWindow;
    } catch (error) {
      log(`cannot read the model catalog for the context window: ${String(error)}`);
    }
    state.catalogContextWindow = { model, size };
    return size;
  }

  /**
   * Retries are transient, so they go out as notices when the client shows
   * them. Otherwise they appear as a line of the agent's thinking, which keeps
   * the user from staring at a silent turn.
   */
  private async reportRetry(
    sessionId: string,
    cx: AgentContext,
    retry: RetryReport,
  ): Promise<void> {
    const text = retryText(retry);
    log(text);
    const update = this.clientFeatures.notices
      ? {
          sessionUpdate: "notice",
          severity: "warning",
          title: text,
        }
      : {
          sessionUpdate: "agent_thought_chunk" as const,
          content: { type: "text" as const, text: `\n${text}\n` },
        };
    await cx.notify(methods.client.session.update, {
      sessionId,
      update: update as never,
    });
  }

  private async startCompaction(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
  ): Promise<string> {
    const id = `compaction-${++state.compactions}`;
    state.openCompactionId = id;
    await cx.notify(methods.client.session.update, {
      sessionId,
      update: this.clientFeatures.compactionUpdates
        ? ({
            sessionUpdate: "compaction_update",
            compactionId: id,
            status: "in_progress",
          } as never)
        : {
            sessionUpdate: "tool_call",
            toolCallId: id,
            title: "Compact context",
            kind: "think",
            status: "in_progress",
          },
    });
    return id;
  }

  /**
   * Shows an image a tool returned. A file-reading tool's image comes from
   * the file itself; any other result that mentions an image waits for the
   * stored message, which holds the image the stream left out.
   */
  private async collectToolImages(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
    toolCallId: string,
    toolCall: StreamedToolCall,
    output: string,
    content: ToolCallContent[],
  ): Promise<void> {
    const path = toolImagePath(toolCall.toolName, toolCall.input.input, state.cwd);
    const image = path ? await readImageFile(path) : undefined;
    if (image) {
      await this.showToolImages(sessionId, cx, toolCallId, content, [image]);
      return;
    }
    if (resultMentionsImage(output)) state.pendingImages.push({ toolCallId, content });
  }

  private async flushPendingImages(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
  ): Promise<void> {
    if (state.pendingImages.length === 0) return;
    const pending = state.pendingImages.splice(0);
    try {
      let messages: readonly unknown[] = [];
      for (let attempt = 0; attempt < 2; attempt++) {
        if (attempt > 0) await new Promise((done) => setTimeout(done, 1_000));
        messages = (await state.session.listMessages({ order: "desc", limit: 30 })).messages;
        if (pending.every((entry) => storedToolImages(messages, entry.toolCallId) !== null)) break;
      }
      for (const entry of pending) {
        const stored = storedToolImages(messages, entry.toolCallId) ?? [];
        if (stored.length === 0) {
          log(`no stored image for tool call ${entry.toolCallId}`);
          continue;
        }
        const images = await Promise.all(
          stored.map((image, index) =>
            this.imageLinks
              ? saveToolImage(image, `${entry.toolCallId}-${index + 1}`)
              : { ...image, name: `image-${index + 1}` },
          ),
        );
        await this.showToolImages(sessionId, cx, entry.toolCallId, entry.content, images);
      }
    } catch (error) {
      log(`could not show tool images: ${String(error)}`);
    }
  }

  /**
   * Adds the images to the tool card as ACP image content. With image links
   * on, they also follow as a markdown image message, which T3 Code renders
   * from the file.
   */
  private async showToolImages(
    sessionId: string,
    cx: AgentContext,
    toolCallId: string,
    content: ToolCallContent[],
    images: ToolImage[],
  ): Promise<void> {
    log(`showing ${images.length} image(s) from tool call ${toolCallId}`);
    await cx.notify(methods.client.session.update, {
      sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        content: [
          ...content,
          ...images.map((image) => ({
            type: "content" as const,
            content: { type: "image" as const, data: image.data, mimeType: image.mimeType },
          })),
        ],
      },
    });
    if (!this.imageLinks) return;
    const markdown = images.map(imageMarkdown).filter(Boolean).join("\n\n");
    if (!markdown) return;
    await cx.notify(methods.client.session.update, {
      sessionId,
      update: {
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: `\n\n${markdown}\n\n` },
      },
    });
  }

  /** Closes a compaction whose summary never arrived. */
  private async settleCompaction(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
  ): Promise<void> {
    if (!state.openCompactionId) return;
    await this.finishCompaction(
      sessionId,
      state,
      { kind: "compaction_summary", summary: "Older messages were summarized to free up context." },
      cx,
    );
  }

  private async finishCompaction(
    sessionId: string,
    state: AcpSessionState,
    summary: CompactionSummary,
    cx: AgentContext,
  ): Promise<void> {
    // Some servers send the summary without announcing the compaction first.
    const id =
      state.openCompactionId ?? (await this.startCompaction(sessionId, state, cx));
    state.openCompactionId = null;
    const text = compactionText(summary);
    await cx.notify(methods.client.session.update, {
      sessionId,
      update: this.clientFeatures.compactionUpdates
        ? ({
            sessionUpdate: "compaction_update",
            compactionId: id,
            status: "completed",
            ...(text ? { summary: [{ type: "text", text }] } : {}),
          } as never)
        : {
            sessionUpdate: "tool_call_update",
            toolCallId: id,
            status: "completed",
            ...(text
              ? {
                  content: [
                    { type: "content" as const, content: { type: "text" as const, text } },
                  ],
                }
              : {}),
          },
    });
  }

  private toPromptResponse(
    state: AcpSessionState,
    result: SDKResultMessage,
  ): PromptResponse {
    if (result.success) {
      return { stopReason: "end_turn" };
    }
    let stopReason: StopReason;
    switch (result.errorCode) {
      case "interrupted":
        stopReason = "cancelled";
        break;
      case "max_steps":
        stopReason = "max_turn_requests";
        break;
      default:
        if (state.cancelled) {
          stopReason = "cancelled";
          break;
        }
        throw new Error(
          result.errorDetail ?? result.error ?? "Letta turn failed",
        );
    }
    return { stopReason };
  }

  /** Letta canUseTool callback -> ACP session/request_permission. */
  private async requestToolPermission(
    sessionId: string,
    toolName: string,
    toolInput: Record<string, unknown>,
    context: CanUseToolContext | undefined,
  ): Promise<CanUseToolResponse> {
    const state = this.sessions.get(sessionId);
    if (!state) {
      return {
        behavior: "deny",
        message: "Unknown ACP session",
      };
    }
    if (state.softCancel) {
      // Between a cancel and the steer that may follow, nothing new runs:
      // with no steer, the cancel was a stop.
      log(`holding ${toolName} until the steer grace period ends`);
      if (!(await state.softCancel.hold)) {
        log(`denying ${toolName}: prompt turn was cancelled`);
        return {
          behavior: "deny",
          message: "Prompt turn was cancelled",
          interrupt: true,
        };
      }
    }
    if (state.cancelled) {
      // The client context now outlives the turn, so nothing else stops a
      // request that raced session/abort from prompting the user again after
      // they pressed stop.
      log(`denying ${toolName}: prompt turn was cancelled`);
      return {
        behavior: "deny",
        message: "Prompt turn was cancelled",
        interrupt: true,
      };
    }
    return this.resolveToolPermission(
      sessionId,
      state,
      state.clientContext,
      toolName,
      toolInput,
      context,
    );
  }

  private async resolveToolPermission(
    sessionId: string,
    state: AcpSessionState,
    cx: AgentContext,
    toolName: string,
    toolInput: Record<string, unknown>,
    context: CanUseToolContext | undefined,
  ): Promise<CanUseToolResponse> {
    const workspaceEditorRead = await editorReadAutoAllows(
      state.modeId,
      toolName,
      toolInput,
      state.cwd,
    );
    if (modeAutoAllows(state.modeId, toolName) || workspaceEditorRead) {
      log(`auto-allowing ${toolName} (mode ${state.modeId})`);
      return { behavior: "allow", updatedInput: toolInput };
    }
    log(
      `permission requested for ${toolName}` +
        (state.promptActive ? "" : " (outside a prompt turn)"),
    );
    if (state.alwaysAllowed.has(toolName)) {
      return { behavior: "allow", updatedInput: toolInput };
    }

    // The approval carries the id of the tool call it belongs to. Without it
    // the client gets a permission card that matches no streamed tool call:
    // guessing by name attaches to the wrong card when several calls are in
    // flight, and the synthetic id orphans the card entirely.
    const toolCallId =
      context?.toolCallId ??
      (state.lastToolCall?.name === toolName
        ? state.lastToolCall.id
        : `${toolName}_${crypto.randomUUID()}`);
    const params = {
      sessionId,
      toolCall: {
        toolCallId,
        title: toolTitle(toolName, toolInput),
        kind: toolKind(toolName),
        status: "pending" as const,
        rawInput: toolInput,
        locations: toolLocations(toolInput),
      },
      options: [
        {
          optionId: "allow_once",
          name: `Allow ${toolName} once`,
          kind: "allow_once" as const,
        },
        {
          optionId: "allow_always",
          name: `Always allow ${toolName} this session`,
          kind: "allow_always" as const,
        },
        {
          optionId: "reject_once",
          name: "Reject",
          kind: "reject_once" as const,
        },
      ],
    };

    let response: RequestPermissionResponse;
    try {
      response = state.promptActive
        ? await cx.request(methods.client.session.requestPermission, params)
        : await withAcpRequestTimeout(
            (signal) =>
              cx.request(methods.client.session.requestPermission, params, {
                cancellationSignal: signal,
              }),
            {
              label: "Permission request",
              description: `approve ${toolName} outside a prompt turn`,
              timeoutMs: this.outOfTurnPermissionTimeoutMs,
            },
          );
    } catch (error) {
      log(`permission request for ${toolName} failed: ${String(error)}`);
      return {
        behavior: "deny",
        message: error instanceof Error ? error.message : String(error),
        interrupt: true,
      };
    }

    if (response.outcome.outcome === "cancelled") {
      return {
        behavior: "deny",
        message: "Prompt turn was cancelled",
        interrupt: true,
      };
    }
    switch (response.outcome.optionId) {
      case "allow_always":
        state.alwaysAllowed.add(toolName);
        return { behavior: "allow", updatedInput: toolInput };
      case "allow_once":
        return { behavior: "allow", updatedInput: toolInput };
      default:
        return { behavior: "deny", message: "User rejected this tool call" };
    }
  }

  private ensureAgent(): Promise<string> {
    if (!this.agentIdPromise) {
      this.agentIdPromise = this.resolveAgent();
      this.agentIdPromise.catch(() => {
        // Allow retry on the next session/new instead of caching the failure.
        this.agentIdPromise = null;
      });
    }
    return this.agentIdPromise;
  }

  private async resolveAgent(): Promise<string> {
    if (this.config.agentId) {
      log(`using existing agent ${this.config.agentId}`);
      return this.config.agentId;
    }
    log("creating a new Letta agent (set LETTA_AGENT_ID to reuse one)...");
    let agentId: string;
    try {
      agentId = await this.client.createAgent({
        // SDK 0.5 made personality presets opt-in. Keep the adapter's established
        // behavior instead of silently creating a zero-memory generic agent.
        personality: "memo",
        name: "Letta",
        description: "Letta agent driven through the Agent Client Protocol",
        model: this.config.model ?? DEFAULT_ACP_MODEL,
      });
    } catch (error) {
      log(
        `agent creation failed: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`,
      );
      throw error;
    }
    log(
      `created agent ${agentId} — set LETTA_AGENT_ID=${agentId} to keep using it`,
    );
    return agentId;
  }
}

/** Run ids an SDK stream message refers to. */
function messageRunIds(message: SDKMessage): string[] {
  const record = message as unknown as Record<string, unknown>;
  const ids: string[] = [];
  if (typeof record.runId === "string") ids.push(record.runId);
  for (const key of ["runIds", "activeRunIds"]) {
    const value = record[key];
    if (Array.isArray(value)) {
      for (const id of value) if (typeof id === "string") ids.push(id);
    }
  }
  return ids;
}

/** LETTA_ACP_TRACE=1 logs every SDK message a prompt consumes (stderr). */
const TRACE = process.env.LETTA_ACP_TRACE === "1";

/** One-line summary of an SDK stream message for tracing (no content). */
function describeSdkMessage(message: SDKMessage): string {
  const record = message as unknown as Record<string, unknown>;
  const fields = ["type", "messageType", "status", "success", "stopReason", "errorCode", "error", "origin", "runIds", "activeRunIds", "runId"]
    .filter((key) => record[key] !== undefined)
    .map((key) => `${key}=${JSON.stringify(record[key])}`);
  const event = record.event as Record<string, unknown> | undefined;
  if (message.type === "stream_event" && event) {
    for (const key of ["message_type", "event_type", "event_data"]) {
      if (event[key] !== undefined) fields.push(`event.${key}=${JSON.stringify(event[key])}`);
    }
  }
  return fields.join(" ").slice(0, 400);
}

/**
 * Model settings the runtime reported when the session started (they carry the
 * active reasoning_effort and context_window). The SDK keeps them but does not
 * expose them publicly yet, so read the field defensively; without it the
 * effort selector falls back to the model's default tier until the user picks.
 */
function runtimeModelSettings(
  session: LettaCodeSession,
): Record<string, unknown> | null | undefined {
  const value: unknown = Reflect.get(session, "_modelSettings");
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function assertConversationId(sessionId: string, action: string): void {
  if (!sessionId.startsWith("conv-") && !sessionId.startsWith("local-conv-")) {
    throw new Error(`Cannot ${action} session ${sessionId}: not a Letta conversation id`);
  }
}

/** Context window of the session's current model, when known. */
function contextWindowSize(state: AcpSessionState): number | undefined {
  const value = (state.currentModelSettings ?? runtimeModelSettings(state.session))
    ?.context_window;
  return typeof value === "number" && value > 0 ? value : undefined;
}

function sessionTitle(conversation: LettaConversation): string | undefined {
  const source = conversation.summary ?? conversation.description;
  if (!source) return undefined;
  const title = source.replace(/[\r\n]+/g, " ").replace(/\s+/g, " ").trim();
  if (!title) return undefined;
  return title.length <= 256 ? title : `${title.slice(0, 255)}…`;
}

/**
 * The SDK reports "?" when a streamed tool_call chunk omits the function name,
 * which every continuation fragment does. Treat that as "unknown" so the name
 * from the opening chunk survives.
 */
function knownToolName(toolName: string): string | undefined {
  if (!toolName || toolName === "?") return undefined;
  return toolName;
}

/** Convert ACP transport shapes to the Agent SDK's keyed MCP configuration. */
export function toSdkMcpServers(
  servers: readonly McpServer[] | undefined,
): McpServers {
  const configs: Array<[string, McpServers[string]]> = [];
  for (const server of servers ?? []) {
    if (!("type" in server)) {
      configs.push([
        server.name,
        {
          command: server.command,
          args: server.args,
          env: Object.fromEntries(
            server.env.map(({ name, value }) => [name, value]),
          ),
        },
      ]);
      continue;
    }
    if (server.type === "http" || server.type === "sse") {
      configs.push([
        server.name,
        {
          type: server.type,
          url: server.url,
          headers: Object.fromEntries(
            server.headers.map(({ name, value }) => [name, value]),
          ),
        },
      ]);
      continue;
    }
    log(`MCP server "${server.name}" skipped: ACP transport is not supported`);
  }
  return Object.fromEntries(configs);
}

/**
 * Whether a prompt can be queued into a running turn: plain text that is not
 * a slash command. Commands run in the adapter or the harness, not the turn.
 */
export function isSteerablePrompt(params: PromptRequest): boolean {
  if (params.prompt.length === 0) return false;
  if (!params.prompt.every((block) => block.type === "text")) return false;
  const first = params.prompt[0];
  return !(first?.type === "text" && first.text.trimStart().startsWith("/"));
}

/** ACP prompt content blocks -> Letta multimodal message content. */
export function toLettaContent(blocks: ContentBlock[]): MessageContentItem[] {
  const content: MessageContentItem[] = [];
  for (const block of blocks) {
    switch (block.type) {
      case "text":
        content.push({ type: "text", text: block.text });
        break;
      case "image":
        if (IMAGE_MEDIA_TYPES.has(block.mimeType)) {
          content.push({
            type: "image",
            source: {
              type: "base64",
              media_type: block.mimeType as
                | "image/png"
                | "image/jpeg"
                | "image/gif"
                | "image/webp",
              data: block.data,
            },
          });
        }
        break;
      case "resource_link":
        content.push({ type: "text", text: `[Referenced file: ${block.uri}]` });
        break;
      case "resource": {
        const resource = block.resource;
        if ("text" in resource && typeof resource.text === "string") {
          content.push({
            type: "text",
            text: `<context uri="${resource.uri}">\n${resource.text}\n</context>`,
          });
        }
        break;
      }
      default:
        // audio and future block types are not advertised in promptCapabilities
        break;
    }
  }
  if (content.length === 0) {
    content.push({ type: "text", text: "" });
  }
  return content;
}

function log(message: string): void {
  // stdout carries the ACP JSON-RPC stream; all logging goes to stderr.
  console.error(`[letta-acp] ${message}`);
}
