import { describe, expect, test } from "bun:test";
import { isPlanTool, PlanTracker } from "../src/plan.js";

describe("plan tracker", () => {
  test("builds the plan from task records and updates", () => {
    const plan = new PlanTracker();
    expect(
      plan.apply("TaskCreate", { subject: "Run tests" }, {
        taskId: "task_1",
        subject: "Run tests",
        status: "pending",
      }),
    ).toBe(true);
    plan.apply("TaskCreate", {}, { taskId: "task_2", subject: "Ship it", status: "pending" });
    expect(
      plan.apply("TaskUpdate", { taskId: "task_1" }, { taskId: "task_1", status: "in_progress" }),
    ).toBe(true);
    expect(plan.entries()).toEqual([
      { content: "Run tests", priority: "medium", status: "in_progress" },
      { content: "Ship it", priority: "medium", status: "pending" },
    ]);
    // Reading a task back changes nothing.
    expect(
      plan.apply("TaskGet", {}, { taskId: "task_2", subject: "Ship it", status: "pending" }),
    ).toBe(false);
    plan.apply("TaskUpdate", {}, { taskId: "task_2", subject: "Ship it", status: "deleted" });
    expect(plan.entries().map((entry) => entry.content)).toEqual(["Run tests"]);
  });

  test("replaces the plan from a task list or todo list", () => {
    const plan = new PlanTracker();
    plan.apply("TaskCreate", {}, { taskId: "task_9", subject: "Old", status: "pending" });
    plan.apply("TaskList", {}, {
      tasks: [
        { taskId: "task_1", subject: "One", status: "completed" },
        { taskId: "task_2", subject: "Two", status: "blocked" },
      ],
    });
    expect(plan.entries()).toEqual([
      { content: "One", priority: "medium", status: "completed" },
      { content: "Two", priority: "medium", status: "pending" },
    ]);
    plan.apply("TodoWrite", { todos: [{ content: "Todo", status: "in_progress" }] }, "ok");
    expect(plan.entries()).toEqual([
      { content: "Todo", priority: "medium", status: "in_progress" },
    ]);
  });

  test("ignores unparseable output and other tools", () => {
    const plan = new PlanTracker();
    expect(plan.apply("TaskCreate", {}, "Created task")).toBe(false);
    expect(plan.apply("Bash", {}, { taskId: "x", subject: "y" })).toBe(false);
    expect(isPlanTool("TaskList")).toBe(true);
    expect(isPlanTool("Read")).toBe(false);
  });
});
