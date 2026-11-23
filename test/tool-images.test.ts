import { describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  imageMarkdown,
  readImageFile,
  resultMentionsImage,
  saveToolImage,
  storedToolImages,
  toolImagePath,
} from "../src/tool-images.js";

describe("tool images", () => {
  test("finds the image file a read tool names", () => {
    expect(toolImagePath("Read", { file_path: "shots/a.PNG" }, "/work")).toBe("/work/shots/a.PNG");
    expect(toolImagePath("ViewImage", { path: "/tmp/b.jpg" }, "/work")).toBe("/tmp/b.jpg");
    expect(toolImagePath("Read", { file_path: "notes.md" }, "/work")).toBeUndefined();
    expect(toolImagePath("Bash", { file_path: "a.png" }, "/work")).toBeUndefined();
  });

  test("recognizes the text Letta leaves for an image", () => {
    expect(resultMentionsImage("[Image: a.png]")).toBe(true);
    expect(resultMentionsImage("done [Image omitted]")).toBe(true);
    expect(resultMentionsImage("Image saved")).toBe(false);
  });

  test("reads image files and skips missing ones", async () => {
    const dir = await mkdtemp(join(tmpdir(), "letta-acp-images-"));
    try {
      const path = join(dir, "pic.webp");
      await writeFile(path, Buffer.from("abc"));
      expect(await readImageFile(path)).toEqual({
        mimeType: "image/webp",
        data: Buffer.from("abc").toString("base64"),
        path,
        name: "pic.webp",
      });
      expect(await readImageFile(join(dir, "missing.png"))).toBeUndefined();
      const saved = await saveToolImage({ mimeType: "image/jpeg", data: "AAAA" }, "call/1", dir);
      expect(saved.path).toBe(join(dir, "call_1.jpg"));
      expect(imageMarkdown(saved)).toBe(`![call_1.jpg](<${join(dir, "call_1.jpg")}>)`);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("reads image parts from a stored tool return", () => {
    const messages = [
      { message_type: "assistant_message", content: "hi" },
      {
        message_type: "tool_return_message",
        tool_call_id: "call-1",
        tool_returns: [
          {
            tool_call_id: "call-1",
            tool_return: [
              { type: "text", text: "[Image: a.png]" },
              { type: "image", source: { type: "base64", media_type: "image/png", data: "QQ==" } },
              { type: "image", source: { type: "letta", file_id: "f", media_type: "image/gif", data: "Qg==" } },
              { type: "image", source: { type: "url", url: "https://example.com/x.png" } },
            ],
          },
        ],
      },
    ];
    expect(storedToolImages(messages, "call-1")).toEqual([
      { mimeType: "image/png", data: "QQ==" },
      { mimeType: "image/gif", data: "Qg==" },
    ]);
    expect(storedToolImages(messages, "call-2")).toBeNull();
  });
});
