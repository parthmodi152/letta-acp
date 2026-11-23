/**
 * Turns Letta's flat model catalog into two ACP session config options: a
 * model selector and a reasoning-effort ("thought_level") selector.
 *
 * Letta's catalog lists one entry per model *and* effort tier, e.g.
 * `gpt-6.2-astra-low`, `gpt-6.2-astra-medium`, `gpt-6.2-astra` (high) all share
 * handle `openai/gpt-6.2-astra`. Letta Code's own model picker groups these by
 * handle and context window and shows effort as a separate control; this
 * module does the same so ACP clients don't show every tier as its own model.
 *
 * Selections still resolve to a single catalog entry id, which is what
 * `session.updateModel()` applies.
 */
import type { SessionConfigOption } from "@agentclientprotocol/sdk";

/** Fields of a Letta `list_models` entry that grouping relies on. */
export interface CatalogEntry {
  id: string;
  handle: string;
  label: string;
  description?: string;
  isDefault?: boolean;
  isFeatured?: boolean;
  updateArgs?: Record<string, unknown>;
}

/** Letta Code's effort order (REASONING_EFFORT_ORDER), lowest to highest. */
export const REASONING_EFFORT_ORDER = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;
export type ReasoningEffort = (typeof REASONING_EFFORT_ORDER)[number];

export const MODEL_CONFIG_ID = "model";
export const REASONING_EFFORT_CONFIG_ID = "reasoning_effort";
/**
 * The effort value of a model without reasoning tiers. Letta Code's picker
 * labels this "Default"; selecting it leaves the model unchanged.
 */
export const DEFAULT_EFFORT_VALUE = "default";

export interface EffortTier {
  effort: ReasoningEffort;
  /** Catalog entry id that applies this tier. */
  entryId: string;
}

export interface ModelGroup {
  /** Stable ACP option value for the group (one of its entry ids). */
  value: string;
  handle: string;
  contextWindow: number | undefined;
  label: string;
  description?: string;
  isDefault: boolean;
  /** Catalog entries in this group, in catalog order. */
  entryIds: string[];
  /** One entry per effort, in REASONING_EFFORT_ORDER. Empty when untiered. */
  tiers: EffortTier[];
}

export interface ModelCatalog {
  groups: ModelGroup[];
  /** Entry id -> its group. */
  groupByEntryId: Map<string, ModelGroup>;
  /** Entry id -> its effort, for entries that have one. */
  effortByEntryId: Map<string, ReasoningEffort>;
}

/** The model and effort a session is using, resolved against the catalog. */
export interface ModelSelection {
  group: ModelGroup;
  /** Undefined when the group has no effort tiers. */
  effort: ReasoningEffort | undefined;
}

export function isReasoningEffort(value: unknown): value is ReasoningEffort {
  return (
    typeof value === "string" &&
    (REASONING_EFFORT_ORDER as readonly string[]).includes(value)
  );
}

function contextWindowOf(entry: { updateArgs?: Record<string, unknown> }) {
  const value = entry.updateArgs?.context_window;
  return typeof value === "number" ? value : undefined;
}

function effortOf(entry: CatalogEntry): ReasoningEffort | undefined {
  const value = entry.updateArgs?.reasoning_effort;
  return isReasoningEffort(value) ? value : undefined;
}

/**
 * Some providers bake the tier into the label ("Gemini 4 Neon (low)"). Strip a
 * trailing "(<effort>)" that matches the entry's own effort.
 */
function baseLabel(entry: CatalogEntry): string {
  const effort = effortOf(entry);
  if (!effort) return entry.label;
  const suffix = ` (${effort})`;
  return entry.label.toLowerCase().endsWith(suffix)
    ? entry.label.slice(0, -suffix.length)
    : entry.label;
}

/**
 * The entry whose id identifies the group: the featured entry, otherwise the
 * shortest id (Letta names the canonical tier without an effort suffix, e.g.
 * `gpt-6.2-astra` next to `gpt-6.2-astra-low`).
 */
function representative(entries: CatalogEntry[]): CatalogEntry {
  const featured = entries.find((entry) => entry.isFeatured);
  if (featured) return featured;
  return entries.reduce((best, entry) =>
    entry.id.length < best.id.length ? entry : best,
  );
}

export function buildModelCatalog(entries: CatalogEntry[]): ModelCatalog {
  const byKey = new Map<string, CatalogEntry[]>();
  const seenIds = new Set<string>();
  for (const entry of entries) {
    if (seenIds.has(entry.id)) continue;
    seenIds.add(entry.id);
    const key = `${entry.handle}\u0000${contextWindowOf(entry) ?? ""}`;
    const members = byKey.get(key);
    if (members) members.push(entry);
    else byKey.set(key, [entry]);
  }

  const groups: ModelGroup[] = [];
  const groupByEntryId = new Map<string, ModelGroup>();
  const effortByEntryId = new Map<string, ReasoningEffort>();
  for (const members of byKey.values()) {
    const rep = representative(members);
    // Like Letta Code's tier builder, the first entry for an effort wins.
    const entryForEffort = new Map<ReasoningEffort, string>();
    for (const entry of members) {
      const effort = effortOf(entry);
      if (!effort) continue;
      effortByEntryId.set(entry.id, effort);
      if (!entryForEffort.has(effort)) entryForEffort.set(effort, entry.id);
    }
    const tiers = REASONING_EFFORT_ORDER.flatMap((effort) => {
      const entryId = entryForEffort.get(effort);
      return entryId ? [{ effort, entryId }] : [];
    });
    const description = members.find((entry) => entry.description)?.description;
    const group: ModelGroup = {
      value: rep.id,
      handle: rep.handle,
      contextWindow: contextWindowOf(rep),
      label: baseLabel(rep),
      ...(description ? { description } : {}),
      isDefault: members.some((entry) => entry.isDefault),
      entryIds: members.map((entry) => entry.id),
      tiers,
    };
    groups.push(group);
    for (const entry of members) groupByEntryId.set(entry.id, group);
  }
  return { groups, groupByEntryId, effortByEntryId };
}

/** Facts known about the session's model, from any of the sources below. */
export interface ModelHints {
  /** Catalog entry id last applied by this adapter (most precise). */
  entryId?: string;
  /** Model id or handle reported by the runtime, LETTA_ACP_MODEL, or /model. */
  model?: string;
  /** Runtime model settings, which may carry reasoning_effort and context_window. */
  settings?: Record<string, unknown> | null;
}

/**
 * Resolves the session's current model and effort. Returns undefined when the
 * model is not in the catalog (e.g. a custom handle); callers present that
 * model as-is with no effort selector.
 */
export function resolveSelection(
  catalog: ModelCatalog,
  hints: ModelHints,
): ModelSelection | undefined {
  for (const id of [hints.entryId, hints.model]) {
    if (!id) continue;
    const group = catalog.groupByEntryId.get(id);
    if (group) {
      const effort = catalog.effortByEntryId.get(id);
      return { group, effort: tierEffort(group, effort) };
    }
  }

  const handle = hints.model;
  if (!handle) return undefined;
  const candidates = catalog.groups.filter((group) => group.handle === handle);
  if (candidates.length === 0) return undefined;
  const contextWindow =
    typeof hints.settings?.context_window === "number"
      ? hints.settings.context_window
      : undefined;
  const group =
    candidates.find((candidate) => candidate.contextWindow === contextWindow) ??
    candidates[0]!;
  const settingsEffort = hints.settings?.reasoning_effort;
  return {
    group,
    effort: tierEffort(
      group,
      isReasoningEffort(settingsEffort)
        ? settingsEffort
        : catalog.effortByEntryId.get(group.value),
    ),
  };
}

/** Keeps an effort only if the group offers it; otherwise the group's default tier. */
function tierEffort(
  group: ModelGroup,
  effort: ReasoningEffort | undefined,
): ReasoningEffort | undefined {
  if (group.tiers.length === 0) return undefined;
  if (effort && group.tiers.some((tier) => tier.effort === effort)) return effort;
  return group.tiers.find((tier) => tier.entryId === group.value)?.effort ??
    group.tiers[0]!.effort;
}

/**
 * The catalog entry to apply when the user picks `groupValue` and/or `effort`.
 * Switching models keeps the current effort when the new model offers it.
 */
export function entryForChoice(
  catalog: ModelCatalog,
  groupValue: string,
  effort: ReasoningEffort | undefined,
): string | undefined {
  const group = catalog.groupByEntryId.get(groupValue);
  if (!group) return undefined;
  if (group.tiers.length === 0) return group.value;
  const tier = effort
    ? group.tiers.find((candidate) => candidate.effort === effort)
    : undefined;
  if (tier) return tier.entryId;
  const fallback = tierEffort(group, undefined);
  return group.tiers.find((candidate) => candidate.effort === fallback)?.entryId;
}

/** Mirrors Letta Code's ModelReasoningSelector labels. */
export function effortLabel(effort: ReasoningEffort, group: ModelGroup): string {
  switch (effort) {
    case "none":
      return "Off";
    case "minimal":
      return "Minimal";
    case "xhigh":
      return group.tiers.some((tier) => tier.effort === "max") ? "Extra High" : "Max";
    case "max":
      return "Max";
    default:
      return effort.charAt(0).toUpperCase() + effort.slice(1);
  }
}

function hasTiers(catalog: ModelCatalog): boolean {
  return catalog.groups.some((group) => group.tiers.length > 0);
}

function effortChoices(group: ModelGroup) {
  return group.tiers.map((tier) => ({
    value: tier.effort,
    name: effortLabel(tier.effort, group),
  }));
}

/**
 * The model selector and, when any catalog model has reasoning tiers, a
 * reasoning-effort selector (ACP category "thought_level"). The effort
 * selector lists the current model's tiers, or a single "Default" choice for a
 * model without tiers: it stays present so clients that show effort per model
 * (T3 Code) keep a reasoning control to attach each model's levels to.
 *
 * A model outside the catalog (custom handle) is kept as the current value so
 * ACP's "currentValue must be an option" rule holds.
 */
export function modelConfigOptions(
  catalog: ModelCatalog,
  selection: ModelSelection | undefined,
  customModel: string | undefined,
): SessionConfigOption[] {
  const options: Array<{ value: string; name: string; description?: string }> =
    catalog.groups.map((group) => ({
      value: group.value,
      name: group.label,
      ...(group.description ? { description: group.description } : {}),
    }));
  let currentValue =
    selection?.group.value ??
    catalog.groups.find((group) => group.isDefault)?.value ??
    catalog.groups[0]?.value;
  if (!selection && customModel) {
    currentValue = customModel;
    options.unshift({ value: customModel, name: customModel });
  }
  if (!currentValue) return [];

  const configOptions: SessionConfigOption[] = [
    {
      id: MODEL_CONFIG_ID,
      name: "Model",
      description: "Model used for this Letta session",
      category: "model",
      type: "select",
      currentValue,
      options,
    },
  ];
  if (hasTiers(catalog)) {
    const group = selection?.group;
    const tiered = group !== undefined && group.tiers.length > 0;
    configOptions.push({
      id: REASONING_EFFORT_CONFIG_ID,
      name: "Reasoning",
      description: tiered
        ? `Reasoning effort for ${group.label}`
        : "This model has no reasoning effort settings",
      category: "thought_level",
      type: "select",
      currentValue: tiered && selection?.effort ? selection.effort : DEFAULT_EFFORT_VALUE,
      options: tiered
        ? effortChoices(group)
        : [{ value: DEFAULT_EFFORT_VALUE, name: "Default" }],
    });
  }
  return configOptions;
}

/** One model in ACP's legacy `models` session state. */
export interface LegacyModelInfo {
  modelId: string;
  name: string;
  description?: string;
  _meta: Record<string, unknown>;
}

/** ACP's legacy (unstable, pre-config-options) `models` session state. */
export interface LegacyModelState {
  currentModelId: string;
  availableModels: LegacyModelInfo[];
}

/**
 * The legacy `models` state, carrying each model's reasoning tiers in `_meta`.
 *
 * Config options only describe the *current* model's effort levels. T3 Code
 * shows effort for every model in its picker by reading these per-model
 * `_meta` fields (`supportsReasoningEffort`, `reasoningEfforts`,
 * `reasoningEffort`), the convention Grok's ACP agent uses. Model ids match the
 * model config option's values, so clients still switch via set_config_option.
 */
export function legacyModelState(
  catalog: ModelCatalog,
  selection: ModelSelection | undefined,
): LegacyModelState | undefined {
  const currentModelId =
    selection?.group.value ??
    catalog.groups.find((group) => group.isDefault)?.value ??
    catalog.groups[0]?.value;
  if (!currentModelId || !hasTiers(catalog)) return undefined;
  return {
    currentModelId,
    availableModels: catalog.groups.map((group) => {
      const effort =
        group === selection?.group ? selection.effort : tierEffort(group, undefined);
      return {
        modelId: group.value,
        name: group.label,
        ...(group.description ? { description: group.description } : {}),
        _meta:
          group.tiers.length > 0 && effort
            ? {
                supportsReasoningEffort: true,
                reasoningEffort: effort,
                reasoningEfforts: effortChoices(group).map((choice) => ({
                  value: choice.value,
                  label: choice.name,
                })),
              }
            : { supportsReasoningEffort: false },
      };
    }),
  };
}
