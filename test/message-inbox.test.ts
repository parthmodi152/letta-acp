import { describe, expect, test } from "bun:test";
import { MessageInbox } from "../src/message-inbox.js";

describe("message inbox", () => {
  test("a timed-out wait does not swallow the next message", async () => {
    const inbox = new MessageInbox<string>();
    expect(await inbox.next(10)).toBeNull();
    inbox.push("late");
    expect(await inbox.next(10)).toEqual({ value: "late", done: false });
  });

  test("delivers queued messages in order, then done once closed", async () => {
    const inbox = new MessageInbox<number>();
    inbox.push(1);
    inbox.push(2);
    const waiting = inbox.next();
    expect(await waiting).toEqual({ value: 1, done: false });
    expect(inbox.drain()).toEqual([2]);
    const pending = inbox.next();
    inbox.close();
    expect(await pending).toEqual({ value: undefined, done: true });
    inbox.push(3);
    expect(await inbox.next()).toEqual({ value: undefined, done: true });
  });
});
