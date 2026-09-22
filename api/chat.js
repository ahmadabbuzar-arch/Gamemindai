// api/chat.js
// Vercel Function using the Web (Fetch API) handler signature — secure
// proxy to Google Gemini (primary) with automatic fallback to Groq
// (backup) if Gemini fails, hits a quota/rate limit, or errors. Keys are
// read only from process.env.GEMINI_API_KEY / process.env.GROQ_API_KEY
// and are never sent to, or exposed in, the frontend.
//
// Using the Web signature (export async function POST(request) with a
// standard Response) is intentional: Vercel's classic Node.js
// (req, res) handler signature does NOT stream by default and needs a
// special account-level flag to force it. The Web signature streams
// out of the box, so the reply appears token-by-token as it's
// generated instead of popping in all at once.

// ---- Gemini (PRIMARY) ----
// gemini-2.5-flash was deprecated/shut down (per Google's own
// deprecations page) — gemini-3.5-flash is the current GA Flash-tier
// model as of this writing. Check
// https://ai.google.dev/gemini-api/docs/models before assuming this
// stays correct long-term; Google rotates Gemini model IDs often.
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash";
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:streamGenerateContent?alt=sse`;

// ---- Groq (FALLBACK) ----
// IMPORTANT: this used to default to "groq/compound", Groq's agentic
// tool-use "system". Groq deprecated compound/compound-mini and
// scheduled them for full shutdown — which is exactly what was causing
// requests (especially ones that triggered its web-search tool, e.g.
// questions about current/outside topics) to silently fail or come back
// empty. Reverted to a plain, currently-supported chat model with no
// tool-use involved, which is far more reliable as a fallback. Verify
// current model status at https://console.groq.com/docs/deprecations.
const GROQ_MODEL = process.env.GROQ_MODEL || "openai/gpt-oss-120b";

// Vision-capable Groq model, used as the vision fallback if Gemini's
// (also multimodal) request fails. qwen/qwen3.6-27b was deprecated in
// favor of qwen/qwen3.8-27b — check
// https://console.groq.com/docs/vision for the current model.
const GROQ_VISION_MODEL = process.env.GROQ_VISION_MODEL || "qwen/qwen3.8-27b";

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";

const UPSTREAM_TIMEOUT_MS = 25000;
const MAX_MESSAGES = 40;
const MAX_MESSAGE_LENGTH = 6000;

const ALLOWED_IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
const MAX_IMAGE_BYTES = 8 * 1024 * 1024; // 8MB decoded

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

// ---- <think>...</think> stripping (Groq path only — some Groq models
// emit visible chain-of-thought by default) ----
function createThinkFilter() {
  let thinkTail = "";
  let insideThink = false;
  return {
    filter(text) {
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
    },
    flush() {
      const out = insideThink ? "" : thinkTail;
      thinkTail = "";
      return out;
    },
  };
}

// Converts our OpenAI-style message list (used for Groq) into Gemini's
// { role, parts } content format. Handles both plain string content and
// the multipart [{type:"text"},{type:"image_url"}] shape used for the
// vision turn.
function toGeminiContents(finalMessages) {
  return finalMessages.map((m) => {
    const role = m.role === "assistant" ? "model" : "user";
    if (Array.isArray(m.content)) {
      const parts = [];
      for (const part of m.content) {
        if (part.type === "text") {
          parts.push({ text: part.text || "" });
        } else if (part.type === "image_url" && part.image_url && part.image_url.url) {
          const match = /^data:([^;]+);base64,(.*)$/.exec(part.image_url.url);
          if (match) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
        }
      }
      return { role, parts: parts.length ? parts : [{ text: "" }] };
    }
    return { role, parts: [{ text: m.content || "" }] };
  });
}

// ---- Attempt Gemini (primary) ----
// Returns a ready-to-stream Response on success, or null on any failure
// (after logging the real reason server-side) so the caller can fall
// back to Groq.
async function attemptGemini(finalMessages, systemPrompt, usingVision) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error("GEMINI_API_KEY is not set — skipping Gemini, falling back to Groq.");
    return null;
  }

  const contents = toGeminiContents(finalMessages);
  const payload = {
    contents,
    systemInstruction: { parts: [{ text: systemPrompt }] },
    generationConfig: {
      temperature: 0.8,
      maxOutputTokens: usingVision ? 1024 : 2048,
    },
  };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let geminiRes;
  try {
    geminiRes = await fetch(GEMINI_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    console.error(`Gemini request failed (network/timeout). model=${GEMINI_MODEL}`, err && err.name, err && err.message);
    return null;
  }

  if (!geminiRes.ok) {
    clearTimeout(timeoutId);
    let data = null;
    try {
      data = await geminiRes.json();
    } catch {
      /* upstream error body wasn't valid JSON */
    }
    const upstreamMessage = (Array.isArray(data) ? data[0] : data)?.error?.message || "";
    // Log the REAL upstream status/reason — never hidden, never shown
    // to the client, but always visible in server logs for debugging.
    console.error(`Gemini API error. model=${GEMINI_MODEL} status=${geminiRes.status} message=${upstreamMessage}`);
    return null; // fall back to Groq
  }

  if (!geminiRes.body) {
    clearTimeout(timeoutId);
    console.error(`Gemini response had no body. model=${GEMINI_MODEL}`);
    return null;
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  let receivedAny = false;

  const stream = new ReadableStream({
    async start(streamController) {
      const reader = geminiRes.body.getReader();
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
            if (!dataStr) continue;

            try {
              const json = JSON.parse(dataStr);
              const candidate = json.candidates && json.candidates[0];
              const parts = candidate && candidate.content && candidate.content.parts;
              if (Array.isArray(parts)) {
                const text = parts.map((p) => (p && typeof p.text === "string" ? p.text : "")).join("");
                if (text) {
                  receivedAny = true;
                  streamController.enqueue(encoder.encode(text));
                }
              }
            } catch {
              /* ignore a malformed SSE chunk and keep reading */
            }
          }
        }
      } catch (err) {
        console.error(`Error while streaming Gemini response. model=${GEMINI_MODEL}`, err);
      } finally {
        clearTimeout(timeoutId);
        if (!receivedAny) {
          console.error(`Gemini stream completed with no content. model=${GEMINI_MODEL}`);
        }
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

// ---- Attempt Groq (fallback) ----
// Returns a ready-to-stream Response on success, or null on failure
// (after logging the real reason) so the caller can return a final,
// clear error to the client.
async function attemptGroq(finalMessages, systemPrompt, usingVision) {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    console.error("GROQ_API_KEY is not set — no fallback available.");
    return null;
  }

  const payload = {
    model: usingVision ? GROQ_VISION_MODEL : GROQ_MODEL,
    messages: [{ role: "system", content: systemPrompt }, ...finalMessages],
    temperature: 0.8,
    max_tokens: usingVision ? 1024 : 2048,
    stream: true,
    // Strip visible chain-of-thought and disable "thinking mode" — both
    // qwen vision and gpt-oss support this; scoped narrowly since not
    // every model recognizes these fields.
    reasoning_format: "hidden",
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
    console.error(`Groq request failed (network/timeout). model=${payload.model}`, err && err.name, err && err.message);
    return null;
  }

  if (!groqRes.ok) {
    clearTimeout(timeoutId);
    let data = null;
    try {
      data = await groqRes.json();
    } catch {
      /* upstream error body wasn't valid JSON */
    }
    const upstreamMessage = (data && data.error && data.error.message) || "";
    console.error(`Groq API error. model=${payload.model} status=${groqRes.status} message=${upstreamMessage}`);
    return null;
  }

  if (!groqRes.body) {
    clearTimeout(timeoutId);
    console.error(`Groq response had no body. model=${payload.model}`);
    return null;
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const thinkFilter = createThinkFilter();
  let receivedAny = false;

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
                const visible = thinkFilter.filter(delta);
                if (visible) {
                  receivedAny = true;
                  streamController.enqueue(encoder.encode(visible));
                }
              }
            } catch {
              /* ignore a malformed SSE chunk and keep reading */
            }
          }
        }
      } catch (err) {
        console.error(`Error while streaming Groq response. model=${payload.model}`, err);
      } finally {
        const remaining = thinkFilter.flush();
        if (remaining) {
          receivedAny = true;
          streamController.enqueue(encoder.encode(remaining));
        }
        clearTimeout(timeoutId);
        if (!receivedAny) {
          console.error(`Groq stream completed with no content. model=${payload.model}`);
        }
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

export async function POST(request) {
  if (!process.env.GEMINI_API_KEY && !process.env.GROQ_API_KEY) {
    console.error("Neither GEMINI_API_KEY nor GROQ_API_KEY is set.");
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
      : "You are GameMind AI, a friendly, natural-sounding general-purpose AI assistant built by Sarim (Sarim Production), with strong gaming expertise. If asked who made you or for a contact email, say Sarim (Sarim Production) and sarimforbusiness@gmail.com — never invent other names or emails. If the user writes in Hindi or Hinglish, reply in casual everyday spoken Hindi/Hinglish (like texting a friend), never shuddh/literary Hindi. Answer directly and briefly — usually 2 to 6 short paragraphs or a few bullet points, no long articles unless the user asks for detail. No emojis.";

  // ---- PRIMARY: Gemini ----
  const geminiResponse = await attemptGemini(finalMessages, systemPrompt, usingVision);
  if (geminiResponse) return geminiResponse;

  // ---- FALLBACK: Groq ----
  console.error("Falling back to Groq after Gemini failed.");
  const groqResponse = await attemptGroq(finalMessages, systemPrompt, usingVision);
  if (groqResponse) return groqResponse;

  // ---- Both providers failed ----
  console.error("Both Gemini and Groq failed for this request.");
  return jsonError("GameMind AI couldn't get a response from any AI provider right now. Please try again in a moment.", 502);
}
