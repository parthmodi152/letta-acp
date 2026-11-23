import { describe, expect, test } from "bun:test";
import {
  buildModelCatalog,
  type CatalogEntry,
  entryForChoice,
  legacyModelState,
  modelConfigOptions,
  REASONING_EFFORT_CONFIG_ID,
  resolveSelection,
} from "../src/model-options.js";

function entry(
  id: string,
  handle: string,
  label: string,
  effort: string | undefined,
  contextWindow: number,
  extra: Partial<CatalogEntry> = {},
): CatalogEntry {
  return {
    id,
    handle,
    label,
    description: `${label} description`,
    updateArgs: {
      context_window: contextWindow,
      ...(effort ? { reasoning_effort: effort } : {}),
    },
    ...extra,
  };
}

// Shapes taken from Letta's live catalog (~/.letta/cache/model-catalog.json).
const CATALOG: CatalogEntry[] = [
  entry("astra-low", "openai/astra", "Astra", "low", 350000),
  entry("astra-medium", "openai/astra", "Astra", "medium", 350000),
  entry("astra", "openai/astra", "Astra", "high", 350000),
  entry("astra-xhigh", "openai/astra", "Astra", "xhigh", 350000),
  entry("astra-max", "openai/astra", "Astra", "max", 350000),
  // Same handle, larger context window: a separate model.
  entry("astra-1m-low", "openai/astra", "Astra 1M", "low", 950000),
  entry("astra-1m", "openai/astra", "Astra 1M", "high", 950000),
  // Untiered model.
  entry("glm", "letta/glm", "Letta GLM", undefined, 200000, { isDefault: true }),
  // A featured alias listed before its duplicate "high" tier.
  entry("opus", "anthropic/opus", "Opus", "high", 200000, { isFeatured: true }),
  entry("opus-low", "anthropic/opus", "Opus", "low", 200000),
  entry("opus-high", "anthropic/opus", "Opus", "high", 200000),
  // Tier baked into the label, and no "max" tier (so xhigh reads "Max").
  entry("neon-low", "google/neon", "Neon (low)", "low", 1048576),
  entry("neon", "google/neon", "Neon", "high", 1048576),
  entry("neon-xhigh", "google/neon", "Neon (xhigh)", "xhigh", 1048576),
];

describe("buildModelCatalog", () => {
  const catalog = buildModelCatalog(CATALOG);

  test("groups tiers by handle and context window", () => {
    expect(catalog.groups.map((group) => [group.value, group.label])).toEqual([
      ["astra", "Astra"],
      ["astra-1m", "Astra 1M"],
      ["glm", "Letta GLM"],
      ["opus", "Opus"],
      ["neon", "Neon"],
    ]);
  });

  test("orders tiers low to high and keeps the first entry per effort", () => {
    const astra = catalog.groupByEntryId.get("astra")!;
    expect(astra.tiers.map((tier) => tier.effort)).toEqual([
      "low",
      "medium",
      "high",
      "xhigh",
      "max",
    ]);
    const opus = catalog.groupByEntryId.get("opus")!;
    expect(opus.tiers).toEqual([
      { effort: "low", entryId: "opus-low" },
      { effort: "high", entryId: "opus" },
    ]);
    expect(catalog.groupByEntryId.get("glm")!.tiers).toEqual([]);
  });
});

describe("resolveSelection", () => {
  const catalog = buildModelCatalog(CATALOG);

  test("an applied entry id pins model and effort", () => {
    const selection = resolveSelection(catalog, { entryId: "astra-medium" });
    expect(selection?.group.value).toBe("astra");
    expect(selection?.effort).toBe("medium");
  });

  test("a handle uses the runtime's context window and effort", () => {
    const selection = resolveSelection(catalog, {
      model: "openai/astra",
      settings: { context_window: 950000, reasoning_effort: "low" },
    });
    expect(selection?.group.value).toBe("astra-1m");
    expect(selection?.effort).toBe("low");
  });

  test("a handle without settings falls back to the canonical tier", () => {
    const selection = resolveSelection(catalog, { model: "openai/astra" });
    expect(selection?.group.value).toBe("astra");
    expect(selection?.effort).toBe("high");
  });

  test("an effort the model lacks falls back to its default tier", () => {
    const selection = resolveSelection(catalog, {
      model: "anthropic/opus",
      settings: { reasoning_effort: "max" },
    });
    expect(selection?.effort).toBe("high");
  });

  test("unknown models are not resolved", () => {
    expect(resolveSelection(catalog, { model: "custom/model" })).toBeUndefined();
  });
});

describe("entryForChoice", () => {
  const catalog = buildModelCatalog(CATALOG);

  test("maps model and effort to a catalog entry", () => {
    expect(entryForChoice(catalog, "astra", "xhigh")).toBe("astra-xhigh");
    expect(entryForChoice(catalog, "astra", "high")).toBe("astra");
  });

  test("switching models keeps the effort when offered, else the default", () => {
    expect(entryForChoice(catalog, "astra-1m", "low")).toBe("astra-1m-low");
    expect(entryForChoice(catalog, "astra-1m", "medium")).toBe("astra-1m");
  });

  test("untiered models ignore effort", () => {
    expect(entryForChoice(catalog, "glm", "high")).toBe("glm");
  });

  test("unknown models are not mapped", () => {
    expect(entryForChoice(catalog, "custom/model", "high")).toBeUndefined();
  });
});

describe("modelConfigOptions", () => {
  const catalog = buildModelCatalog(CATALOG);

  test("lists each model once and adds a thought_level option", () => {
    const options = modelConfigOptions(
      catalog,
      resolveSelection(catalog, { entryId: "astra-medium" }),
      "openai/astra",
    );
    expect(options).toHaveLength(2);
    const [model, effort] = options;
    expect(model).toMatchObject({
      id: "model",
      category: "model",
      currentValue: "astra",
    });
    expect(
      (model as { options: Array<{ value: string }> }).options.map(
        (option) => option.value,
      ),
    ).toEqual(["astra", "astra-1m", "glm", "opus", "neon"]);
    expect(effort).toEqual({
      id: REASONING_EFFORT_CONFIG_ID,
      name: "Reasoning",
      description: "Reasoning effort for Astra",
      category: "thought_level",
      type: "select",
      currentValue: "medium",
      options: [
        { value: "low", name: "Low" },
        { value: "medium", name: "Medium" },
        { value: "high", name: "High" },
        { value: "xhigh", name: "Extra High" },
        { value: "max", name: "Max" },
      ],
    });
  });

  test("labels xhigh as Max when the model has no max tier", () => {
    const [, effort] = modelConfigOptions(
      catalog,
      resolveSelection(catalog, { entryId: "neon-xhigh" }),
      undefined,
    );
    expect(
      (effort as { options: Array<{ value: string; name: string }> }).options,
    ).toEqual([
      { value: "low", name: "Low" },
      { value: "high", name: "High" },
      { value: "xhigh", name: "Max" },
    ]);
  });

  test("offers a single Default effort for untiered models", () => {
    const options = modelConfigOptions(
      catalog,
      resolveSelection(catalog, { entryId: "glm" }),
      "letta/glm",
    );
    expect(options[1]).toMatchObject({
      id: REASONING_EFFORT_CONFIG_ID,
      category: "thought_level",
      currentValue: "default",
      options: [{ value: "default", name: "Default" }],
    });
  });

  test("omits the effort option when no model has tiers", () => {
    const untiered = buildModelCatalog([CATALOG[7]!]);
    const options = modelConfigOptions(
      untiered,
      resolveSelection(untiered, { entryId: "glm" }),
      "letta/glm",
    );
    expect(options.map((option) => option.id)).toEqual(["model"]);
  });

  test("describes every model's tiers in the legacy models state", () => {
    const state = legacyModelState(
      catalog,
      resolveSelection(catalog, { entryId: "astra-medium" }),
    );
    expect(state?.currentModelId).toBe("astra");
    const byId = new Map(state!.availableModels.map((model) => [model.modelId, model]));
    expect([...byId.keys()]).toEqual(["astra", "astra-1m", "glm", "opus", "neon"]);
    // The current model reports its applied effort...
    expect(byId.get("astra")!._meta).toMatchObject({
      supportsReasoningEffort: true,
      reasoningEffort: "medium",
    });
    // ...other models their default tier.
    expect(byId.get("opus")!._meta).toEqual({
      supportsReasoningEffort: true,
      reasoningEffort: "high",
      reasoningEfforts: [
        { value: "low", label: "Low" },
        { value: "high", label: "High" },
      ],
    });
    expect(byId.get("glm")!._meta).toEqual({ supportsReasoningEffort: false });
  });

  test("keeps a custom model as the current value", () => {
    const [model] = modelConfigOptions(catalog, undefined, "custom/model");
    expect(model).toMatchObject({ currentValue: "custom/model" });
    expect(
      (model as { options: Array<{ value: string; name: string }> }).options[0],
    ).toEqual({ value: "custom/model", name: "custom/model" });
  });
});
