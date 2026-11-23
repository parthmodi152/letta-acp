import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, extname, isAbsolute, join, resolve } from "node:path";

/**
 * Images that tools return to the model.
 *
 * Letta Code streams only the text of a tool result (for an image, a label
 * such as `[Image: chart.png]`), but the model gets the image, and the stored
 * message keeps it in `tool_returns[].tool_return`. The adapter recovers the
 * image from the file the tool read, or else from the stored message.
 */

export interface ToolImage {
  mimeType: string;
  /** Base64 image data. */
  data: string;
  /** File on this machine that holds the image, for clients that link to it. */
  path?: string;
  name: string;
}

const MIME_BY_EXTENSION: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
};
const EXTENSION_BY_MIME: Record<string, string> = {
  "image/png": ".png",
  "image/jpeg": ".jpg",
  "image/gif": ".gif",
  "image/webp": ".webp",
};

/** Largest image the adapter embeds in an ACP update. */
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

/** The streamed text Letta Code leaves where a tool returned an image. */
const IMAGE_MARKER = /\[Image(?::| omitted\])/;

export function resultMentionsImage(content: string): boolean {
  return IMAGE_MARKER.test(content);
}

/** The image file a file-reading tool read, when its input names one. */
export function toolImagePath(
  toolName: string,
  input: Record<string, unknown> | undefined,
  cwd: string,
): string | undefined {
  if (!input) return undefined;
  const raw =
    toolName === "Read"
      ? input.file_path
      : toolName === "ViewImage"
        ? (input.path ?? input.file_path)
        : undefined;
  if (typeof raw !== "string" || !raw.trim()) return undefined;
  if (!MIME_BY_EXTENSION[extname(raw).toLowerCase()]) return undefined;
  return isAbsolute(raw) ? raw : resolve(cwd, raw);
}

/** Reads an image file, or undefined when it is missing, too large or not an image. */
export async function readImageFile(path: string): Promise<ToolImage | undefined> {
  const mimeType = MIME_BY_EXTENSION[extname(path).toLowerCase()];
  if (!mimeType) return undefined;
  try {
    const info = await stat(path);
    if (!info.isFile() || info.size > MAX_IMAGE_BYTES) return undefined;
    const data = (await readFile(path)).toString("base64");
    return { mimeType, data, path, name: basename(path) };
  } catch {
    return undefined;
  }
}

/**
 * Image parts of the stored tool return for `toolCallId`, or null when the
 * message is not stored yet.
 */
export function storedToolImages(
  messages: readonly unknown[],
  toolCallId: string,
): Omit<ToolImage, "name">[] | null {
  for (const message of messages) {
    const record = asRecord(message);
    if (!record || record.message_type !== "tool_return_message") continue;
    const returns = Array.isArray(record.tool_returns) ? record.tool_returns : [record];
    for (const entry of returns) {
      const toolReturn = asRecord(entry);
      if (!toolReturn) continue;
      const id = toolReturn.tool_call_id ?? record.tool_call_id;
      if (id !== toolCallId) continue;
      const parts = Array.isArray(toolReturn.tool_return) ? toolReturn.tool_return : [];
      return parts.flatMap((part) => {
        const image = asRecord(part);
        if (image?.type !== "image") return [];
        const source = asRecord(image.source);
        const data = typeof source?.data === "string" ? source.data : undefined;
        const mimeType =
          typeof source?.media_type === "string" ? source.media_type : undefined;
        if (!data || !mimeType || !EXTENSION_BY_MIME[mimeType]) return [];
        if (data.length > (MAX_IMAGE_BYTES * 4) / 3) return [];
        return [{ mimeType, data }];
      });
    }
  }
  return null;
}

/** Saves an image that has no file yet, so clients can link to it. */
export async function saveToolImage(
  image: Omit<ToolImage, "name">,
  baseName: string,
  dir = join(homedir(), ".letta", "letta-acp", "images"),
): Promise<ToolImage> {
  const name = `${baseName.replace(/[^A-Za-z0-9_-]/g, "_")}${EXTENSION_BY_MIME[image.mimeType] ?? ".img"}`;
  await mkdir(dir, { recursive: true });
  const path = join(dir, name);
  await writeFile(path, Buffer.from(image.data, "base64"));
  return { ...image, path, name };
}

/** Markdown that shows the image from its file, e.g. `![chart.png](</tmp/chart.png>)`. */
export function imageMarkdown(image: ToolImage): string {
  if (!image.path) return "";
  const alt = image.name.replace(/[[\]]/g, "");
  return `![${alt}](<${image.path.replace(/[<>]/g, "")}>)`;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}
