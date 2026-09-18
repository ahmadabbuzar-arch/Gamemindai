// api/generate-image.js
// Vercel Function (Web/Fetch API handler signature) — secure proxy to
// Cloudflare Workers AI's image-generation REST API. Credentials are
// read only from process.env.CLOUDFLARE_ACCOUNT_ID and
// process.env.CLOUDFLARE_API_TOKEN, and are never sent to, or exposed
// in, the frontend. Plain REST call — no SDK/npm install required.

const CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell";

// flux-1-schnell's documented input schema is just { prompt, steps } —
// it has NO width/height/aspect_ratio/negative_prompt parameter. Sending
// unsupported fields (e.g. width/height) causes Cloudflare to reject the
// whole request, not just ignore the extra field. So aspect ratio, style,
// and negative prompt are all folded into the prompt text instead, the
// same way a person would type them.
const UPSTREAM_TIMEOUT_MS = 45000; // image generation is slower than a text reply
const MAX_PROMPT_LENGTH = 2048; // Cloudflare's own documented cap for this model
const MAX_NEGATIVE_PROMPT_LENGTH = 500;
const DIFFUSION_STEPS = 4; // Cloudflare's own default for this model; max is 8

const ALLOWED_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4"];

// flux-1-schnell has no aspect-ratio parameter, so this is expressed as a
// plain-language composition hint appended to the prompt instead.
const ASPECT_RATIO_HINTS = {
  "1:1": "square 1:1 composition",
  "16:9": "wide 16:9 landscape composition",
  "9:16": "tall 9:16 vertical portrait composition",
  "4:3": "4:3 landscape composition",
  "3:4": "3:4 portrait composition",
};

// Short descriptor phrases folded into the prompt text — same approach
// used for aspect ratio, since this model has no separate "style" param.
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

// Strips the raw token/account id out of any upstream message before it
// is ever shown to the browser — defense in depth in case Cloudflare
// ever echoes a header value back in an error body.
function sanitizeUpstreamMessage(message, ...secrets) {
  if (!message) return "";
  let safe = String(message);
  for (const secret of secrets) {
    if (secret) safe = safe.split(secret).join("[redacted]");
  }
  return safe.slice(0, 300);
}

export async function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

export async function POST(request) {
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
  const apiToken = process.env.CLOUDFLARE_API_TOKEN;

  if (!accountId || !apiToken) {
    console.error("CLOUDFLARE_ACCOUNT_ID and/or CLOUDFLARE_API_TOKEN is not set.");
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

  const cleanAspectRatio = ALLOWED_ASPECT_RATIOS.includes(aspectRatio) ? aspectRatio : "1:1";
  const aspectHint = ASPECT_RATIO_HINTS[cleanAspectRatio];

  const cleanNegative =
    typeof negativePrompt === "string" ? negativePrompt.trim().slice(0, MAX_NEGATIVE_PROMPT_LENGTH) : "";

  const styleDescriptor = typeof style === "string" && STYLE_DESCRIPTORS[style] ? STYLE_DESCRIPTORS[style] : "";

  // Build the suffix first so the user's own prompt — the important
  // part — is what gets truncated last if the combined text is too long.
  let suffix = `. ${aspectHint}.`;
  if (styleDescriptor) suffix += ` Style: ${styleDescriptor}.`;
  if (cleanNegative) suffix += ` Do not include: ${cleanNegative}.`;

  const cleanPrompt = prompt.trim().slice(0, Math.max(0, MAX_PROMPT_LENGTH - suffix.length));
  const finalPrompt = (cleanPrompt + suffix).slice(0, MAX_PROMPT_LENGTH);

  // Request structure: only "prompt" and "steps" are sent — this model
  // does not accept width/height/aspect_ratio/negative_prompt fields.
  const payload = {
    prompt: finalPrompt,
    steps: DIFFUSION_STEPS,
  };

  const CLOUDFLARE_API_URL = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${CLOUDFLARE_MODEL}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let cfRes;
  try {
    cfRes = await fetch(CLOUDFLARE_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      console.error(`Cloudflare image request timed out. model=${CLOUDFLARE_MODEL}`);
      return jsonError("Image generation took too long. Please try again.", 504);
    }
    console.error(`Unexpected error calling Cloudflare Workers AI. model=${CLOUDFLARE_MODEL}`, err);
    return jsonError("Couldn't reach the image service. Check your connection and try again.", 500);
  }

  clearTimeout(timeoutId);

  let data = null;
  try {
    data = await cfRes.json();
  } catch {
    console.error(
      `Cloudflare returned an unreadable (non-JSON) response. model=${CLOUDFLARE_MODEL} status=${cfRes.status}`
    );
    return jsonError("Received an unreadable response from the image service.", 502);
  }

  const cfErrors = Array.isArray(data && data.errors) ? data.errors : [];
  const firstError = cfErrors[0];
  const rawMessage = (firstError && firstError.message) || "";
  const errorCode = firstError && firstError.code;
  const safeMessage = sanitizeUpstreamMessage(rawMessage, apiToken, accountId);

  // Cloudflare's /ai/run endpoint signals failure either via a non-2xx
  // HTTP status or via `success: false` in an otherwise 200 response —
  // check both rather than assuming only one applies.
  if (!cfRes.ok || data.success === false) {
    const status = cfRes.status;

    console.error(
      `Cloudflare image API error. model=${CLOUDFLARE_MODEL} httpStatus=${status} errorCode=${errorCode || "none"} message=${rawMessage}`
    );

    // 401 / 403 — invalid token, wrong account id, or missing Workers AI
    // permission on the token.
    if (status === 401 || status === 403) {
      return jsonError(
        `The image service rejected the request (invalid credentials or missing permissions)${
          safeMessage ? `: ${safeMessage}` : "."
        } Please check CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID.`,
        status
      );
    }

    // 429 — Workers AI daily/rate quota exceeded.
    if (status === 429) {
      return jsonError(
        `Image generation quota or rate limit reached${safeMessage ? `: ${safeMessage}` : "."} ` +
          "Please wait a moment and try again.",
        429
      );
    }

    // 400 — invalid request (bad prompt, unsupported field, etc.).
    if (status === 400) {
      return jsonError(
        `The image request was invalid${safeMessage ? `: ${safeMessage}` : "."} Try adjusting your prompt.`,
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

  const base64 = data && data.result && data.result.image;

  if (!base64 || typeof base64 !== "string") {
    console.error(
      `Cloudflare response had no image data. model=${CLOUDFLARE_MODEL} body=${JSON.stringify(data).slice(0, 500)}`
    );
    return jsonError("The AI didn't return an image. Please try rephrasing your prompt.", 502);
  }

  // flux-1-schnell's documented output is a base64-encoded JPEG.
  return new Response(JSON.stringify({ imageDataUrl: `data:image/jpeg;base64,${base64}` }), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  });
}
