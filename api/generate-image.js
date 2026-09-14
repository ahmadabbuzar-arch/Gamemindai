// api/generate-image.js
// Vercel Function (Web/Fetch API handler signature) — secure proxy to the
// Gemini image-generation API. The key is read only from
// process.env.GEMINI_API_KEY and is never sent to, or exposed in, the
// frontend. Plain REST calls are used — no SDK/npm install required.

const GEMINI_MODEL = "gemini-2.5-flash-image";
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
      return jsonError("Image generation took too long. Please try again.", 504);
    }
    console.error("Unexpected error calling Gemini API:", err);
    return jsonError("Couldn't reach the image service. Check your connection and try again.", 500);
  }

  clearTimeout(timeoutId);

  let data = null;
  try {
    data = await geminiRes.json();
  } catch {
    return jsonError("Received an unreadable response from the image service.", 502);
  }

  if (!geminiRes.ok) {
    const status = geminiRes.status;
    const upstreamMessage = (data && data.error && data.error.message) || "";
    console.error("Gemini API error:", status, upstreamMessage);

    if (status === 400 && /api[_ ]?key/i.test(upstreamMessage)) {
      return jsonError("The image generation API key looks invalid. Please check the server configuration.", 500);
    }
    if (status === 401 || status === 403) {
      return jsonError("The image generation API key was rejected. Please check the server configuration.", 500);
    }
    if (status === 429) {
      return jsonError("Image generation rate limit reached. Please wait a moment and try again.", 429);
    }
    if (status === 400) {
      return jsonError("That request couldn't be processed. Try adjusting your prompt.", 400);
    }
    return jsonError("The image service returned an error. Please try again.", 502);
  }

  // Content blocked for safety before any generation happened.
  const blockReason = data && data.promptFeedback && data.promptFeedback.blockReason;
  if (blockReason) {
    return jsonError("That prompt was blocked by the safety filter. Please try a different description.", 422);
  }

  const candidate = data && data.candidates && data.candidates[0];
  const finishReason = candidate && candidate.finishReason;
  if (finishReason === "SAFETY" || finishReason === "PROHIBITED_CONTENT") {
    return jsonError("That request was blocked by the safety filter. Please try a different description.", 422);
  }

  const parts = candidate && candidate.content && candidate.content.parts;
  const imagePart = Array.isArray(parts) ? parts.find((p) => p && p.inlineData && p.inlineData.data) : null;

  if (!imagePart) {
    console.error("Gemini response had no image data:", JSON.stringify(data).slice(0, 500));
    return jsonError("The AI didn't return an image. Please try rephrasing your prompt.", 502);
  }

  const mimeType = imagePart.inlineData.mimeType || "image/png";
  const base64 = imagePart.inlineData.data;

  return new Response(JSON.stringify({ imageDataUrl: `data:${mimeType};base64,${base64}` }), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  });
}
