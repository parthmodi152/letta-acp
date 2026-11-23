import type {
  CreateElicitationResponse,
  ElicitationPropertySchema,
  ElicitationSchema,
} from "@agentclientprotocol/sdk";

/**
 * Letta's AskUserQuestion tool, shown as an ACP form elicitation.
 *
 * The tool is approved like any other, and the answers travel back in the
 * approval: `updatedInput.answers` maps each question's text to the chosen
 * label (or the user's own words), which is how Letta's own channels answer it.
 *
 * Each question becomes a choice field `q<n>` plus an optional free-text
 * companion `q<n>__other`. T3 Code folds that pair into one question with a
 * custom-answer box; other clients show the text field separately.
 */

export const ASK_USER_QUESTION_TOOL = "AskUserQuestion";

interface AskQuestion {
  question: string;
  header?: string;
  options: Array<{ label: string; description?: string }>;
  multiSelect: boolean;
}

export interface AskUserForm {
  message: string;
  requestedSchema: ElicitationSchema;
  questions: AskQuestion[];
}

/** Builds the form for an AskUserQuestion input, or null when it has no usable questions. */
export function askUserForm(input: Record<string, unknown>): AskUserForm | null {
  const questions = parseQuestions(input.questions);
  if (questions.length === 0) return null;

  const properties: Record<string, ElicitationPropertySchema> = {};
  questions.forEach((question, index) => {
    const id = fieldId(index);
    const title = question.header || `Question ${index + 1}`;
    if (question.options.length === 0) {
      properties[id] = { type: "string", title, description: question.question };
      return;
    }
    const choices = question.options.map((option) => ({
      const: option.label,
      title: option.label,
      ...(option.description ? { description: option.description } : {}),
    }));
    properties[id] = question.multiSelect
      ? {
          type: "array",
          title,
          description: question.question,
          items: { anyOf: choices },
        }
      : {
          type: "string",
          title,
          description: question.question,
          oneOf: choices,
        };
    properties[`${id}__other`] = {
      type: "string",
      title: `${title}: other answer`,
      description: "Type your own answer instead",
    };
  });

  return {
    message:
      questions.length === 1
        ? questions[0]!.question
        : "The agent has a few questions for you.",
    requestedSchema: { type: "object", properties, required: [] },
    questions,
  };
}

export type AskUserOutcome =
  | { kind: "answered"; answers: Record<string, string> }
  | { kind: "declined"; message: string };

/** Turns the client's form response into AskUserQuestion answers. */
export function askUserAnswers(
  form: AskUserForm,
  response: CreateElicitationResponse,
): AskUserOutcome {
  if (response.action !== "accept") {
    return {
      kind: "declined",
      message:
        response.action === "decline"
          ? "The user declined to answer these questions."
          : "The user dismissed these questions without answering.",
    };
  }
  const content: Record<string, unknown> =
    (("content" in response ? response.content : null) as Record<string, unknown> | null) ?? {};
  const answers: Record<string, string> = {};
  form.questions.forEach((question, index) => {
    const id = fieldId(index);
    const parts: string[] = [];
    const selected = content[id];
    if (typeof selected === "string" && selected.trim()) parts.push(selected.trim());
    if (Array.isArray(selected)) {
      for (const item of selected) {
        if (typeof item === "string" && item.trim()) parts.push(item.trim());
      }
    }
    const other = content[`${id}__other`];
    if (typeof other === "string" && other.trim()) parts.push(other.trim());
    if (parts.length > 0) answers[question.question] = parts.join(", ");
  });
  if (Object.keys(answers).length === 0) {
    return { kind: "declined", message: "The user submitted no answers." };
  }
  return { kind: "answered", answers };
}

function fieldId(index: number): string {
  return `q${index + 1}`;
}

function parseQuestions(value: unknown): AskQuestion[] {
  if (!Array.isArray(value)) return [];
  const questions: AskQuestion[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") continue;
    const raw = item as Record<string, unknown>;
    const question = typeof raw.question === "string" ? raw.question.trim() : "";
    if (!question) continue;
    const options = Array.isArray(raw.options)
      ? raw.options.flatMap((option) => {
          if (!option || typeof option !== "object") return [];
          const { label, description } = option as Record<string, unknown>;
          if (typeof label !== "string" || !label.trim()) return [];
          return [
            {
              label: label.trim(),
              ...(typeof description === "string" && description.trim()
                ? { description: description.trim() }
                : {}),
            },
          ];
        })
      : [];
    // Labels are the answer values, so they must be unique.
    const unique = options.filter(
      (option, index) => options.findIndex((o) => o.label === option.label) === index,
    );
    questions.push({
      question,
      ...(typeof raw.header === "string" && raw.header.trim()
        ? { header: raw.header.trim() }
        : {}),
      options: unique,
      multiSelect: raw.multiSelect === true,
    });
  }
  return questions;
}
