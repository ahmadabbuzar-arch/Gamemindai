// api/chat.js
// Vercel Function using the Web (Fetch API) handler signature — secure
// proxy to the Groq API. The API key is read only from
// process.env.GROQ_API_KEY and is never sent to, or exposed in, the
// frontend.
//
// Using the Web signature (export async function POST(request) with a
// standard Response) is intentional: Vercel's classic Node.js
// (req, res) handler signature does NOT stream by default and needs a
// special account-level flag to force it. The Web signature streams
// out of the box, so the reply appears token-by-token as it's
// generated instead of popping in all at once.

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";

// Change this to swap models without touching any other code.
const GROQ_MODEL = process.env.GROQ_MODEL || "openai/gpt-oss-120b";

// Vision-capable model, used only for messages that include an image.
// Groq's vision lineup changes more often than its text lineup — check
// https://console.groq.com/docs/vision for the current model before
// depending on this long-term.
const GROQ_VISION_MODEL = process.env.GROQ_VISION_MODEL || "qwen/qwen3.6-27b";

const UPSTREAM_TIMEOUT_MS = 25000;
const MAX_MESSAGES = 40;
const MAX_MESSAGE_LENGTH = 6000;

const ALLOWED_IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
const MAX_IMAGE_BYTES = 8 * 1024 * 1024; // 8MB decoded — well under Groq's 20MB request limit

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function jsonError(message, status) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  });
}

// The browser sends a CORS preflight (OPTIONS) request before the real
// POST whenever the page calling this API is on a different origin —
// which is exactly the case when this endpoint is called from inside
// an APK/WebView wrapper. Without this handler, that preflight fails
// and the real request never goes out.
export async function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

export async function POST(request) {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    console.error("GROQ_API_KEY is not set.");
    return jsonError("The server isn't configured yet. Please try again later.", 500);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return jsonError("Invalid request body.", 400);
  }

  const { messages, system, image } = body || {};

  if (!Array.isArray(messages) || messages.length === 0) {
    return jsonError("No message provided.", 400);
  }
  if (messages.length > MAX_MESSAGES) {
    return jsonError("This conversation is too long. Please start a new chat.", 400);
  }

  const cleanedMessages = [];
  for (const m of messages) {
    if (!m || typeof m.content !== "string") continue;
    const role = m.role === "assistant" ? "assistant" : "user";
    cleanedMessages.push({ role, content: m.content.slice(0, MAX_MESSAGE_LENGTH) });
  }
  // Drop empty text-only messages, but keep the very last one even if
  // empty — it may carry only an image with no caption text.
  while (cleanedMessages.length > 1 && !cleanedMessages[0].content.trim()) {
    cleanedMessages.shift();
  }

  if (cleanedMessages.length === 0) {
    return jsonError("No valid message content provided.", 400);
  }

  const hasImagePayload = image != null;
  if (!hasImagePayload && !cleanedMessages[cleanedMessages.length - 1].content.trim()) {
    return jsonError("No message provided.", 400);
  }

  // ---- Image validation (optional) ----
  let imageDataUrl = null;
  let usingVision = false;

  if (image != null) {
    if (typeof image !== "object" || typeof image.mimeType !== "string" || typeof image.data !== "string") {
      return jsonError("Invalid image data.", 400);
    }
    if (!ALLOWED_IMAGE_MIME_TYPES.includes(image.mimeType)) {
      return jsonError("Please attach a PNG, JPEG, or WebP image.", 400);
    }
    if (!image.data.trim()) {
      return jsonError("Invalid image data.", 400);
    }

    let byteLength;
    try {
      byteLength = Buffer.from(image.data, "base64").length;
    } catch {
      return jsonError("Invalid image data.", 400);
    }
    if (byteLength === 0) {
      return jsonError("Invalid image data.", 400);
    }
    if (byteLength > MAX_IMAGE_BYTES) {
      return jsonError("That image is too large. Please attach one under 8MB.", 400);
    }

    imageDataUrl = `data:${image.mimeType};base64,${image.data}`;
    usingVision = true;
  }

  const lastMessage = cleanedMessages[cleanedMessages.length - 1];
  const finalMessages = usingVision
    ? [
        ...cleanedMessages.slice(0, -1),
        {
          role: lastMessage.role,
          content: [
            { type: "text", text: lastMessage.content || "What can you tell me about this image?" },
            { type: "image_url", image_url: { url: imageDataUrl } },
          ],
        },
      ]
    : cleanedMessages;

  const systemPrompt =
    typeof system === "string" && system.trim()
      ? system.trim().slice(0, 2000)
      : "You are GameMind AI, a friendly, natural-sounding general-purpose AI assistant with strong gaming expertise. If the user writes in Hindi or Hinglish, reply in casual everyday spoken Hindi/Hinglish (like texting a friend), never shuddh/literary Hindi. Answer directly and briefly — usually 2 to 6 short paragraphs or a few bullet points, no long articles unless the user asks for detail. No emojis.";

  const payload = {
    model: usingVision ? GROQ_VISION_MODEL : GROQ_MODEL,
    messages: [{ role: "system", content: systemPrompt }, ...finalMessages],
    temperature: 0.8,
    max_tokens: 1024,
    stream: true,
    // Some models (including the vision model) emit their chain-of-thought
    // as visible <think>...</think> text by default — "hidden" strips that
    // so only the final answer is ever streamed to the user.
    reasoning_format: "hidden",
    // The vision model defaults to "thinking mode", which produces long,
    // slow, essay-length answers even for simple image questions. Force
    // its efficient non-thinking dialogue mode instead.
    ...(usingVision ? { reasoning_effort: "none" } : {}),
  };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let groqRes;
  try {
    groqRes = await fetch(GROQ_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      return jsonError("The AI service took too long to respond. Please try again.", 504);
    }
    console.error("Unexpected error calling Groq API:", err);
    return jsonError("Something went wrong on our end. Please try again.", 500);
  }

  if (!groqRes.ok) {
    clearTimeout(timeoutId);
    let data = null;
    try {
      data = await groqRes.json();
    } catch {
      /* upstream error body wasn't valid JSON — fall through to generic message */
    }
    const upstreamMessage =
      data && data.error && data.error.message ? data.error.message : "The AI service returned an error.";
    console.error("Groq API error:", groqRes.status, upstreamMessage);
    return jsonError("GameMind AI couldn't get a response right now. Please try again.", 502);
  }

  if (!groqRes.body) {
    clearTimeout(timeoutId);
    return jsonError("GameMind AI couldn't get a response right now. Please try again.", 502);
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  // Safety net: strip any <think>...</think> block that slips through
  // even with reasoning_format: "hidden" (e.g. a future model change).
  // Handles the tags arriving split across multiple stream chunks.
  let thinkTail = "";
  let insideThink = false;
  function filterThinking(text) {
    thinkTail += text;
    let output = "";
    while (true) {
      if (!insideThink) {
        const openIdx = thinkTail.indexOf("<think>");
        if (openIdx === -1) {
          const safeLen = Math.max(0, thinkTail.length - 6);
          output += thinkTail.slice(0, safeLen);
          thinkTail = thinkTail.slice(safeLen);
          break;
        }
        output += thinkTail.slice(0, openIdx);
        thinkTail = thinkTail.slice(openIdx + 7);
        insideThink = true;
      } else {
        const closeIdx = thinkTail.indexOf("</think>");
        if (closeIdx === -1) {
          const safeLen = Math.max(0, thinkTail.length - 7);
          thinkTail = thinkTail.slice(safeLen);
          break;
        }
        thinkTail = thinkTail.slice(closeIdx + 8);
        insideThink = false;
      }
    }
    return output;
  }
  function flushThinking() {
    const out = insideThink ? "" : thinkTail;
    thinkTail = "";
    return out;
  }

  // Re-package Groq's SSE stream ("data: {...}\n\n" lines) into plain
  // text token chunks, so the browser can just read and append text
  // without needing to know anything about the SSE/OpenAI format.
  const stream = new ReadableStream({
    async start(streamController) {
      const reader = groqRes.body.getReader();
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine.startsWith("data:")) continue;
            const dataStr = trimmedLine.slice(5).trim();
            if (!dataStr || dataStr === "[DONE]") continue;

            try {
              const json = JSON.parse(dataStr);
              const delta = json.choices && json.choices[0] && json.choices[0].delta && json.choices[0].delta.content;
              if (delta) {
                const visible = filterThinking(delta);
                if (visible) streamController.enqueue(encoder.encode(visible));
              }
            } catch {
              /* ignore a malformed SSE chunk and keep reading */
            }
          }
        }
      } catch (err) {
        console.error("Error while streaming Groq response:", err);
      } finally {
        const remaining = flushThinking();
        if (remaining) streamController.enqueue(encoder.encode(remaining));
        clearTimeout(timeoutId);
        streamController.close();
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      ...CORS_HEADERS,
    },
  });
}
