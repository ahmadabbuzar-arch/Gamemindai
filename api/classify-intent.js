// api/classify-intent.js
// Vercel Function (Web/Fetch API handler signature) — a fast, cheap Groq
// call that classifies a chat message so the frontend can automatically
// route it to normal chat, image generation, image editing, or image
// analysis — without the user ever having to manually pick a mode.
// Uses the existing GROQ_API_KEY; no new credentials required.

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
// A small, fast model is enough for classification — no need for the
// heavier compound/vision models used elsewhere.
const CLASSIFY_MODEL = process.env.GROQ_CLASSIFY_MODEL || "openai/gpt-oss-20b";

const UPSTREAM_TIMEOUT_MS = 12000;
const MAX_MESSAGE_LENGTH = 2000;
const MAX_CONTEXT_MESSAGES = 6;
const MAX_CONTEXT_CHARS = 1500;

const ALLOWED_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4"];
const ALLOWED_INTENTS = ["CHAT", "IMAGE_GENERATE", "IMAGE_EDIT", "IMAGE_ANALYZE"];

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

function fallbackResult(message) {
  // Never hard-fail the chat flow over a classification hiccup — default
  // to plain CHAT with the original message untouched.
  return new Response(JSON.stringify({ intent: "CHAT", prompt: message, aspectRatio: null }), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  });
}

export async function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

export async function POST(request) {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    console.error("GROQ_API_KEY is not set (needed by classify-intent too).");
    return jsonError("Not configured yet.", 500);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return jsonError("Invalid request body.", 400);
  }

  const { message, imageAvailable, recentContext } = body || {};

  if (typeof message !== "string" || !message.trim()) {
    return jsonError("No message provided.", 400);
  }
  const cleanMessage = message.trim().slice(0, MAX_MESSAGE_LENGTH);
  const hasImage = !!imageAvailable;

  let contextBlock = "";
  if (Array.isArray(recentContext)) {
    const recent = recentContext.slice(-MAX_CONTEXT_MESSAGES);
    contextBlock = recent
      .map((m) => `${m && m.role === "user" ? "User" : "Assistant"}: ${String((m && m.content) || "").slice(0, 300)}`)
      .join("\n")
      .slice(0, MAX_CONTEXT_CHARS);
  }

  const systemPrompt =
    "You classify one chat message for an AI app that can chat, generate images, edit images, and analyze images. " +
    "Reply with ONLY a JSON object: {\"intent\": one of CHAT, IMAGE_GENERATE, IMAGE_EDIT, IMAGE_ANALYZE, \"prompt\": a string, \"aspectRatio\": one of \"1:1\",\"16:9\",\"9:16\",\"4:3\",\"3:4\",\"4:5\",\"5:4\" or null}. " +
    "Rules:\n" +
    "- IMAGE_GENERATE: the user wants a brand-new image created from a description, and no existing image needs to change. If they say something like \"turn this prompt into an image\" or \"make this into an image\" referring to earlier text in the conversation, use that earlier text (not the short instruction itself) to build the \"prompt\".\n" +
    "- IMAGE_EDIT: the user wants to modify, fix, or change an existing image (colors, text, background, style, \"make it better\", \"isko purple kar do\", etc.) — only choose this if an image is currently available (see imageAvailable below). If they ask for an edit-sounding change but no image is available, use CHAT instead so the assistant can ask them to attach one.\n" +
    "- IMAGE_ANALYZE: the user is asking a question ABOUT an available image (what does it say, describe it, what's wrong with it) without asking to change it.\n" +
    "- CHAT: anything else — normal conversation, questions, requests unrelated to creating/editing an image.\n" +
    "For IMAGE_GENERATE and IMAGE_EDIT, write \"prompt\" as a clean, standalone instruction for an image model, resolving vague references (\"this\", \"it\", \"the logo\") using the conversation context you're given. For CHAT and IMAGE_ANALYZE, just repeat the user's original message as \"prompt\". " +
    "If the user mentions a size/ratio (square, widescreen, portrait, story, Instagram post, 16:9, 4:5, etc.), set \"aspectRatio\" to the closest match, else null. " +
    "The user may write in English, Hindi, or Hinglish (romanized Hindi) — classify correctly regardless of language. " +
    "Output ONLY the JSON object, nothing else — no explanation, no markdown fences.";

  const userPrompt =
    `imageAvailable: ${hasImage}\n` +
    (contextBlock ? `Recent conversation:\n${contextBlock}\n\n` : "") +
    `New message: ${cleanMessage}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let res;
  try {
    res = await fetch(GROQ_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: CLASSIFY_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        temperature: 0,
        max_tokens: 300,
        stream: false,
        response_format: { type: "json_object" },
      }),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    console.error("classify-intent: network error calling Groq:", err && err.name);
    return fallbackResult(cleanMessage);
  }
  clearTimeout(timeoutId);

  if (!res.ok) {
    let errData = null;
    try {
      errData = await res.json();
    } catch {
      /* ignore */
    }
    console.error("classify-intent: Groq API error", res.status, JSON.stringify(errData).slice(0, 300));
    return fallbackResult(cleanMessage);
  }

  let data;
  try {
    data = await res.json();
  } catch {
    console.error("classify-intent: unreadable Groq response");
    return fallbackResult(cleanMessage);
  }

  const raw = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
  let parsed = null;
  try {
    parsed = raw ? JSON.parse(raw) : null;
  } catch {
    console.error("classify-intent: model did not return valid JSON:", String(raw).slice(0, 200));
    return fallbackResult(cleanMessage);
  }

  if (!parsed || !ALLOWED_INTENTS.includes(parsed.intent)) {
    return fallbackResult(cleanMessage);
  }

  // Never allow IMAGE_EDIT to come back when no image is actually
  // available — enforce this server-side too, not just via the prompt.
  let intent = parsed.intent;
  if (intent === "IMAGE_EDIT" && !hasImage) {
    intent = "CHAT";
  }

  const promptOut =
    typeof parsed.prompt === "string" && parsed.prompt.trim() ? parsed.prompt.trim().slice(0, 2000) : cleanMessage;
  const aspectRatioOut = ALLOWED_ASPECT_RATIOS.includes(parsed.aspectRatio) ? parsed.aspectRatio : null;

  return new Response(JSON.stringify({ intent, prompt: promptOut, aspectRatio: aspectRatioOut }), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  });
}
