import type { PlanEntry } from "@agentclientprotocol/sdk";

/**
 * Tracks the agent's task list so it can be shown as an ACP plan.
 *
 * Letta Code keeps progress in its task tools: each returns the full task list
 * as `tasks` (TaskCreate, TaskUpdate and TaskGet also return the record they
 * touched). Other toolsets send the whole list as tool input: TodoWrite /
 * todo_write `todos`, or UpdatePlan `plan`. ACP's `plan` update replaces the client's checklist each time, so the
 * tracker keeps the full list and reports it whenever it changes.
 */

type PlanStatus = PlanEntry["status"];

interface PlanTask {
  content: string;
  status: PlanStatus;
}

const TASK_RECORD_TOOLS = new Set(["TaskCreate", "TaskUpdate", "TaskGet"]);
const TODO_LIST_TOOLS = new Set(["TodoWrite", "todo_write", "write_todos"]);
/** Codex-style toolsets send the whole plan as `plan: [{ step, status }]`. */
const PLAN_LIST_TOOLS = new Set(["UpdatePlan", "update_plan"]);

export function isPlanTool(toolName: string): boolean {
  return (
    TASK_RECORD_TOOLS.has(toolName) ||
    TODO_LIST_TOOLS.has(toolName) ||
    PLAN_LIST_TOOLS.has(toolName) ||
    toolName === "TaskList"
  );
}

export class PlanTracker {
  private tasks = new Map<string, PlanTask>();

  /**
   * Applies a successful plan-tool call. Returns true when the plan changed
   * and should be sent again.
   */
  apply(
    toolName: string,
    input: Record<string, unknown> | undefined,
    output: unknown,
  ): boolean {
    const before = this.signature();
    if (TODO_LIST_TOOLS.has(toolName)) {
      const todos = input?.todos;
      if (!Array.isArray(todos)) return false;
      this.replace(todos);
    } else if (PLAN_LIST_TOOLS.has(toolName)) {
      const steps = input?.plan;
      if (!Array.isArray(steps)) return false;
      this.replace(steps);
    } else if (toolName === "TaskList" || TASK_RECORD_TOOLS.has(toolName)) {
      // Every task tool returns the full list as `tasks`, which also covers
      // deletions. A bare record is the fallback for older harnesses.
      const result = record(output);
      if (Array.isArray(result?.tasks)) this.replace(result.tasks);
      else if (result && toolName !== "TaskList") this.upsert(result);
      else return false;
    } else {
      return false;
    }
    return this.signature() !== before;
  }

  entries(): PlanEntry[] {
    return [...this.tasks.values()].map((task) => ({
      content: task.content,
      priority: "medium",
      status: task.status,
    }));
  }

  private replace(items: unknown[]): void {
    this.tasks.clear();
    items.forEach((item, index) => {
      const task = record(item);
      if (!task) return;
      const id = stringField(task, ["taskId", "id"]) ?? `item-${index}`;
      const parsed = planTask(task);
      if (parsed) this.tasks.set(id, parsed);
    });
  }

  private upsert(task: Record<string, unknown>): void {
    const id = stringField(task, ["taskId", "id"]);
    if (!id) return;
    if (task.status === "deleted") {
      this.tasks.delete(id);
      return;
    }
    const parsed = planTask(task, this.tasks.get(id));
    if (parsed) this.tasks.set(id, parsed);
  }

  private signature(): string {
    return JSON.stringify([...this.tasks.entries()]);
  }
}

function planTask(
  task: Record<string, unknown>,
  previous?: PlanTask,
): PlanTask | undefined {
  const content =
    stringField(task, ["subject", "content", "step", "description"]) ??
    previous?.content;
  if (!content) return undefined;
  return {
    content,
    status: planStatus(task.status) ?? previous?.status ?? "pending",
  };
}

function planStatus(value: unknown): PlanStatus | undefined {
  switch (value) {
    case "pending":
    case "in_progress":
    case "completed":
      return value;
    default:
      return undefined;
  }
}

function stringField(
  value: Record<string, unknown>,
  keys: string[],
): string | undefined {
  for (const key of keys) {
    const field = value[key];
    if (typeof field === "string" && field.trim()) return field.trim();
  }
  return undefined;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}
