// api/generate-image.js
// Vercel Function (Web/Fetch API handler signature) — secure proxy to the
// Gemini image-generation API. The key is read only from
// process.env.GEMINI_API_KEY and is never sent to, or exposed in, the
// frontend. Plain REST calls are used — no SDK/npm install required.

// gemini-2.5-flash-image is on Google's own deprecation schedule for
// shutdown on October 2, 2026 (released Oct 2, 2025); Google's listed
// recommended replacement is gemini-3.1-flash-image-preview, which is
// what current image-generation docs default to. Check
// https://ai.google.dev/gemini-api/docs/deprecations before assuming
// this stays correct long-term — Google rotates image model IDs often.
//
// IMPORTANT: Gemini image-generation models have NO free tier — Google's
// pricing page lists "Free Tier: not available" for every current image
// model. The Google Cloud project behind GEMINI_API_KEY must have
// billing enabled, or every request will fail with a quota/permission
// error regardless of how little you use it.
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.1-flash-image-preview";
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

const UPSTREAM_TIMEOUT_MS = 45000; // image generation is slower than a text reply
const MAX_PROMPT_LENGTH = 2000;
const MAX_NEGATIVE_PROMPT_LENGTH = 500;

const ALLOWED_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4"];

// Short descriptor phrases folded into the prompt text — Gemini's image
// model doesn't have a separate "style" API parameter, so style is
// expressed the same way a person would type it.
const STYLE_DESCRIPTORS = {
  Realistic: "photorealistic, natural lighting, high detail",
  Cinematic: "cinematic lighting, dramatic composition, film still, wide dynamic range",
  Anime: "anime style, vibrant cel-shaded illustration, Japanese animation art",
  "3D Render": "3D render, octane render, soft studio lighting, detailed materials",
  "Pixel Art": "pixel art, retro video game style, crisp pixels, limited color palette",
  Fantasy: "fantasy art, epic and imaginative, painterly detail",
  Horror: "horror atmosphere, eerie and unsettling, moody dark lighting",
  Minimalist: "minimalist, clean simple shapes, lots of negative space",
};

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

// Strips anything that looks like the raw API key value out of an
// upstream message before it's ever shown to the browser — defense in
// depth in case Google ever echoes a key/header back in an error body.
function sanitizeUpstreamMessage(message, apiKey) {
  if (!message) return "";
  let safe = String(message);
  if (apiKey) safe = safe.split(apiKey).join("[redacted]");
  return safe.slice(0, 300);
}

export async function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

export async function POST(request) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error("GEMINI_API_KEY is not set.");
    return jsonError("Image generation isn't configured yet. Please try again later.", 500);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return jsonError("Invalid request body.", 400);
  }

  const { prompt, style, aspectRatio, negativePrompt } = body || {};

  if (typeof prompt !== "string" || !prompt.trim()) {
    return jsonError("Please enter a prompt to generate an image.", 400);
  }
  const cleanPrompt = prompt.trim().slice(0, MAX_PROMPT_LENGTH);

  const cleanAspectRatio = ALLOWED_ASPECT_RATIOS.includes(aspectRatio) ? aspectRatio : "1:1";

  const cleanNegative =
    typeof negativePrompt === "string" ? negativePrompt.trim().slice(0, MAX_NEGATIVE_PROMPT_LENGTH) : "";

  const styleDescriptor = typeof style === "string" && STYLE_DESCRIPTORS[style] ? STYLE_DESCRIPTORS[style] : "";

  let finalPrompt = cleanPrompt;
  if (styleDescriptor) finalPrompt += `. Style: ${styleDescriptor}.`;
  if (cleanNegative) finalPrompt += ` Do not include: ${cleanNegative}.`;

  // Request structure and image-output handling preserved as-is —
  // only the model id and error handling below have changed.
  const payload = {
    contents: [{ parts: [{ text: finalPrompt }] }],
    generationConfig: {
      responseModalities: ["IMAGE"],
      imageConfig: { aspectRatio: cleanAspectRatio },
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
    if (err.name === "AbortError") {
      console.error(`Gemini image request timed out. model=${GEMINI_MODEL}`);
      return jsonError("Image generation took too long. Please try again.", 504);
    }
    console.error(`Unexpected error calling Gemini API. model=${GEMINI_MODEL}`, err);
    return jsonError("Couldn't reach the image service. Check your connection and try again.", 500);
  }

  clearTimeout(timeoutId);

  let data = null;
  try {
    data = await geminiRes.json();
  } catch {
    console.error(`Gemini returned an unreadable (non-JSON) response. model=${GEMINI_MODEL} status=${geminiRes.status}`);
    return jsonError("Received an unreadable response from the image service.", 502);
  }

  if (!geminiRes.ok) {
    const status = geminiRes.status;
    const rawMessage = (data && data.error && data.error.message) || "";
    const upstreamStatus = (data && data.error && data.error.status) || "";
    const safeMessage = sanitizeUpstreamMessage(rawMessage, apiKey);

    // Safe server-side logging for debugging — never sent to the client.
    console.error(
      `Gemini image API error. model=${GEMINI_MODEL} httpStatus=${status} upstreamStatus=${upstreamStatus} message=${rawMessage}`
    );

    // 429 — rate limit or quota exceeded. Image models have no free
    // tier, so this often means billing isn't enabled rather than a
    // transient spike — say so rather than just "try again."
    if (status === 429) {
      return jsonError(
        `Image generation quota or rate limit reached${safeMessage ? `: ${safeMessage}` : "."} ` +
          "Gemini image models have no free tier — check that billing is enabled on your Google Cloud project, or wait and try again if you're already on a paid plan.",
        429
      );
    }

    // 401 / 403 — API key, permission, or billing problem.
    if (status === 401 || status === 403) {
      return jsonError(
        `The image service rejected the request (API key, permission, or billing issue)${
          safeMessage ? `: ${safeMessage}` : "."
        } Please check the Gemini API key and billing configuration.`,
        status
      );
    }

    // 400 — invalid request or an unsupported/unknown model id.
    if (status === 400) {
      return jsonError(
        `The image request was invalid${safeMessage ? `: ${safeMessage}` : "."} ` +
          "This can happen if the configured model name is no longer supported — try adjusting your prompt, or check the server's GEMINI_MODEL setting.",
        400
      );
    }

    // Anything else — surface the real upstream status and message
    // instead of a generic, misleading one.
    return jsonError(
      `The image service returned an error (status ${status})${safeMessage ? `: ${safeMessage}` : "."}`,
      502
    );
  }

  // Content blocked for safety before any generation happened.
  const blockReason = data && data.promptFeedback && data.promptFeedback.blockReason;
  if (blockReason) {
    console.error(`Gemini blocked the prompt before generation. model=${GEMINI_MODEL} blockReason=${blockReason}`);
    return jsonError("That prompt was blocked by the safety filter. Please try a different description.", 422);
  }

  const candidate = data && data.candidates && data.candidates[0];
  const finishReason = candidate && candidate.finishReason;
  if (finishReason === "SAFETY" || finishReason === "PROHIBITED_CONTENT") {
    console.error(`Gemini blocked the output. model=${GEMINI_MODEL} finishReason=${finishReason}`);
    return jsonError("That request was blocked by the safety filter. Please try a different description.", 422);
  }

  const parts = candidate && candidate.content && candidate.content.parts;
  const imagePart = Array.isArray(parts) ? parts.find((p) => p && p.inlineData && p.inlineData.data) : null;

  if (!imagePart) {
    console.error(
      `Gemini response had no image data. model=${GEMINI_MODEL} finishReason=${finishReason || "none"} body=${JSON.stringify(data).slice(0, 500)}`
    );
    return jsonError("The AI didn't return an image. Please try rephrasing your prompt.", 502);
  }

  const mimeType = imagePart.inlineData.mimeType || "image/png";
  const base64 = imagePart.inlineData.data;

  return new Response(JSON.stringify({ imageDataUrl: `data:${mimeType};base64,${base64}` }), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  });
}
