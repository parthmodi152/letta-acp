import { describe, expect, test } from "bun:test";
import { askUserAnswers, askUserForm } from "../src/ask-user.js";

const input = {
  questions: [
    {
      question: "Which library should we use?",
      header: "Library",
      options: [
        { label: "date-fns", description: "Small and modular" },
        { label: "luxon", description: "Time zones built in" },
      ],
    },
    {
      question: "Which checks should run?",
      header: "Checks",
      multiSelect: true,
      options: [
        { label: "lint", description: "" },
        { label: "test", description: "" },
      ],
    },
  ],
};

describe("AskUserQuestion forms", () => {
  test("maps questions to choice fields with free-text companions", () => {
    const form = askUserForm(input)!;
    expect(form.message).toBe("The agent has a few questions for you.");
    expect(form.requestedSchema.properties).toEqual({
      q1: {
        type: "string",
        title: "Library",
        description: "Which library should we use?",
        oneOf: [
          { const: "date-fns", title: "date-fns", description: "Small and modular" },
          { const: "luxon", title: "luxon", description: "Time zones built in" },
        ],
      },
      q1__other: {
        type: "string",
        title: "Library: other answer",
        description: "Type your own answer instead",
      },
      q2: {
        type: "array",
        title: "Checks",
        description: "Which checks should run?",
        items: {
          anyOf: [
            { const: "lint", title: "lint" },
            { const: "test", title: "test" },
          ],
        },
      },
      q2__other: {
        type: "string",
        title: "Checks: other answer",
        description: "Type your own answer instead",
      },
    });
    expect(form.requestedSchema.required).toEqual([]);
  });

  test("answers by question text, joining selections and custom text", () => {
    const form = askUserForm(input)!;
    expect(
      askUserAnswers(form, {
        action: "accept",
        content: { q1__other: "dayjs", q2: ["lint", "test"] },
      }),
    ).toEqual({
      kind: "answered",
      answers: {
        "Which library should we use?": "dayjs",
        "Which checks should run?": "lint, test",
      },
    });
  });

  test("treats decline, cancel and empty answers as no answer", () => {
    const form = askUserForm(input)!;
    expect(askUserAnswers(form, { action: "decline" }).kind).toBe("declined");
    expect(askUserAnswers(form, { action: "cancel" }).kind).toBe("declined");
    expect(askUserAnswers(form, { action: "accept", content: {} }).kind).toBe("declined");
  });

  test("asks option-less questions as plain text and skips empty input", () => {
    const form = askUserForm({ questions: [{ question: "What name?", header: "Name", options: [] }] })!;
    expect(form.message).toBe("What name?");
    expect(form.requestedSchema.properties).toEqual({
      q1: { type: "string", title: "Name", description: "What name?" },
    });
    expect(askUserForm({ questions: [] })).toBeNull();
  });
});
