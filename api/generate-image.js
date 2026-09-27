// api/generate-image.js
// Vercel Function (Web/Fetch API handler signature) — secure proxy to
// Cloudflare Workers AI's image generation AND image editing REST APIs.
// Credentials are read only from process.env.CLOUDFLARE_ACCOUNT_ID and
// process.env.CLOUDFLARE_API_TOKEN, and are never sent to, or exposed
// in, the frontend. Plain REST calls — no SDK/npm install required.

// Text-to-image (no source image attached): stable-diffusion-xl-base-1.0.
// ROOT CAUSE OF THE "always square" BUG: this used to be
// flux-1-schnell, whose input schema is ONLY { prompt, steps } — it has
// no width/height/aspect_ratio parameter at all. Every request was
// silently generated at Flux's default square resolution no matter what
// aspect ratio the UI sent, because there was nowhere to actually pass
// it to the provider — the old code could only fold a text hint like
// "16:9 landscape composition" into the prompt, which influences
// framing/composition but not the actual output dimensions. SDXL's
// schema genuinely accepts width/height (256-2048px each) plus a real
// negative_prompt field, so the selected ratio can now be sent as real
// generation dimensions instead of just a prompt suggestion. Slower
// ("Medium" vs Flux's "Fast" per Cloudflare's own model guide) but this
// is what it takes to actually honor the selected aspect ratio.
const CLOUDFLARE_GENERATE_MODEL = "@cf/stabilityai/stable-diffusion-xl-base-1.0";

// SDXL's own required range is 256-2048px per side. Pairs below keep
// each ratio close to SDXL's ~1024x1024 trained resolution (bucketed,
// rounded to multiples of 8) rather than stretching/cropping after the
// fact — this IS the aspect ratio at the generation stage, not a
// post-hoc crop.
const ASPECT_RATIO_DIMENSIONS = {
  "1:1": { width: 1024, height: 1024 },
  "16:9": { width: 1344, height: 768 },
  "9:16": { width: 768, height: 1344 },
  "4:3": { width: 1152, height: 896 },
  "3:4": { width: 896, height: 1152 },
  "4:5": { width: 912, height: 1144 },
  "5:4": { width: 1144, height: 912 },
};

// SDXL's documented default/max is 20 — there's no lower "fast" tier
// like Flux's 4-step schnell, so this is fixed rather than a tunable
// speed/quality knob.
const SDXL_NUM_STEPS = 20;

// Image editing (a source image is attached): flux-2-klein-4b — this is
// the current Cloudflare model that actually accepts an input image for
// editing/reference. Its request format is multipart/form-data with a
// "prompt" field and up to 4 "input_image_N" binary fields, and it
// requires every input image to be smaller than 512x512.
const CLOUDFLARE_EDIT_MODEL = "@cf/black-forest-labs/flux-2-klein-4b";

const UPSTREAM_TIMEOUT_MS = 45000; // image generation is slower than a text reply
const MAX_PROMPT_LENGTH = 2048; // Conservative shared cap across the Cloudflare text-to-image models used here
const MAX_NEGATIVE_PROMPT_LENGTH = 500;
const MAX_EDIT_IMAGE_BYTES = 3 * 1024 * 1024; // decoded size guard for the 512x512 input limit

const ALLOWED_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4"];

// Neither model has an aspect-ratio parameter, so this is expressed as a
// plain-language composition hint appended to the prompt instead.
const ASPECT_RATIO_HINTS = {
  "1:1": "square 1:1 composition",
  "16:9": "wide 16:9 landscape composition",
  "9:16": "tall 9:16 vertical portrait composition",
  "4:3": "4:3 landscape composition",
  "3:4": "3:4 portrait composition",
  "4:5": "4:5 vertical portrait composition (Instagram post style)",
  "5:4": "5:4 landscape composition",
};

// Short descriptor phrases folded into the prompt text — same approach
// used for aspect ratio, since these models have no separate "style" param.
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

const ALLOWED_EDIT_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];

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

// Style has no dedicated API parameter on either Cloudflare model, so
// it's still folded into the prompt text. Aspect ratio is now handled
// for real via width/height (see ASPECT_RATIO_DIMENSIONS) — the text
// hint stays only as an extra compositional nudge, not the mechanism
// that controls output dimensions. Negative prompt is no longer folded
// in here: SDXL has a real negative_prompt field (see below), so it's
// sent as its own parameter instead of prompt text.
function buildPromptSuffix(cleanAspectRatio, style) {
  const aspectHint = ASPECT_RATIO_HINTS[cleanAspectRatio];
  const styleDescriptor = typeof style === "string" && STYLE_DESCRIPTORS[style] ? STYLE_DESCRIPTORS[style] : "";

  let suffix = aspectHint ? `. ${aspectHint}.` : "";
  if (styleDescriptor) suffix += ` Style: ${styleDescriptor}.`;
  return suffix;
}

// Shared error categorization for both the generate and edit Cloudflare
// calls — same response envelope, same failure modes, same messages.
function cloudflareErrorResponse(cfRes, data, modelName, secrets) {
  const cfErrors = Array.isArray(data && data.errors) ? data.errors : [];
  const firstError = cfErrors[0];
  const rawMessage = (firstError && firstError.message) || "";
  const errorCode = firstError && firstError.code;
  const safeMessage = sanitizeUpstreamMessage(rawMessage, ...secrets);
  const status = cfRes.status;

  console.error(
    `Cloudflare image API error. model=${modelName} httpStatus=${status} errorCode=${errorCode || "none"} message=${rawMessage}`
  );

  if (status === 401 || status === 403) {
    return jsonError(
      `The image service rejected the request (invalid credentials or missing permissions)${
        safeMessage ? `: ${safeMessage}` : "."
      } Please check CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID.`,
      status
    );
  }

  if (status === 429) {
    return jsonError(
      `Image generation quota or rate limit reached${safeMessage ? `: ${safeMessage}` : "."} ` +
        "Please wait a moment and try again.",
      429
    );
  }

  if (status === 400) {
    return jsonError(
      `The image request was invalid${safeMessage ? `: ${safeMessage}` : "."} Try adjusting your prompt${
        modelName === CLOUDFLARE_EDIT_MODEL ? " or using a smaller/simpler source image" : ""
      }.`,
      400
    );
  }

  return jsonError(
    `The image service returned an error (status ${status})${safeMessage ? `: ${safeMessage}` : "."}`,
    502
  );
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

  const { prompt, style, aspectRatio, negativePrompt, image } = body || {};

  if (typeof prompt !== "string" || !prompt.trim()) {
    return jsonError("Please enter a prompt to generate an image.", 400);
  }

  const cleanAspectRatio = ALLOWED_ASPECT_RATIOS.includes(aspectRatio) ? aspectRatio : "1:1";
  const suffix = buildPromptSuffix(cleanAspectRatio, style);
  const cleanPrompt = prompt.trim().slice(0, Math.max(0, MAX_PROMPT_LENGTH - suffix.length));
  const finalPrompt = (cleanPrompt + suffix).slice(0, MAX_PROMPT_LENGTH);
  const cleanNegativePrompt =
    typeof negativePrompt === "string" ? negativePrompt.trim().slice(0, MAX_NEGATIVE_PROMPT_LENGTH) : "";

  const isEdit = image != null;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let cfRes;

  if (isEdit) {
    // ---- Image editing path: flux-2-klein-4b, multipart/form-data ----
    if (typeof image !== "object" || typeof image.mimeType !== "string" || typeof image.data !== "string") {
      clearTimeout(timeoutId);
      return jsonError("Invalid source image data.", 400);
    }
    if (!ALLOWED_EDIT_IMAGE_TYPES.includes(image.mimeType)) {
      clearTimeout(timeoutId);
      return jsonError("The source image must be a PNG, JPEG, or WebP.", 400);
    }

    let imageBuffer;
    try {
      imageBuffer = Buffer.from(image.data, "base64");
    } catch {
      clearTimeout(timeoutId);
      return jsonError("Invalid source image data.", 400);
    }
    if (!imageBuffer || imageBuffer.length === 0) {
      clearTimeout(timeoutId);
      return jsonError("Invalid source image data.", 400);
    }
    if (imageBuffer.length > MAX_EDIT_IMAGE_BYTES) {
      clearTimeout(timeoutId);
      return jsonError(
        "That source image is too large for editing. Please use a smaller image (under 512x512 works best).",
        400
      );
    }

    const form = new FormData();
    form.append("prompt", finalPrompt);
    form.append("input_image_0", new Blob([imageBuffer], { type: image.mimeType }), "source." + (image.mimeType.split("/")[1] || "png"));

    const CLOUDFLARE_API_URL = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${CLOUDFLARE_EDIT_MODEL}`;

    try {
      cfRes = await fetch(CLOUDFLARE_API_URL, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiToken}` }, // no Content-Type — fetch sets the multipart boundary itself
        body: form,
        signal: controller.signal,
      });
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        console.error(`Cloudflare image-edit request timed out. model=${CLOUDFLARE_EDIT_MODEL}`);
        return jsonError("Image editing took too long. Please try again.", 504);
      }
      console.error(`Unexpected error calling Cloudflare Workers AI (edit). model=${CLOUDFLARE_EDIT_MODEL}`, err);
      return jsonError("Couldn't reach the image service. Check your connection and try again.", 500);
    }
  } else {
    // ---- Text-to-image generation path: stable-diffusion-xl-base-1.0,
    // JSON body. This is the fix for the aspect-ratio bug: width/height
    // are real, honored parameters on this model (unlike flux-1-schnell,
    // which had nowhere to receive them), so the selected ratio reaches
    // the provider as actual generation dimensions rather than being
    // silently dropped and defaulting to a square image. negative_prompt
    // is likewise sent as its own field instead of being folded into the
    // prompt text.
    const dimensions = ASPECT_RATIO_DIMENSIONS[cleanAspectRatio] || ASPECT_RATIO_DIMENSIONS["1:1"];
    const payload = {
      prompt: finalPrompt,
      width: dimensions.width,
      height: dimensions.height,
      num_steps: SDXL_NUM_STEPS,
      ...(cleanNegativePrompt ? { negative_prompt: cleanNegativePrompt } : {}),
    };
    const CLOUDFLARE_API_URL = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${CLOUDFLARE_GENERATE_MODEL}`;

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
        console.error(`Cloudflare image request timed out. model=${CLOUDFLARE_GENERATE_MODEL}`);
        return jsonError("Image generation took too long. Please try again.", 504);
      }
      console.error(`Unexpected error calling Cloudflare Workers AI. model=${CLOUDFLARE_GENERATE_MODEL}`, err);
      return jsonError("Couldn't reach the image service. Check your connection and try again.", 500);
    }
  }

  clearTimeout(timeoutId);

  const modelUsed = isEdit ? CLOUDFLARE_EDIT_MODEL : CLOUDFLARE_GENERATE_MODEL;

  let data = null;
  try {
    data = await cfRes.json();
  } catch {
    console.error(`Cloudflare returned an unreadable (non-JSON) response. model=${modelUsed} status=${cfRes.status}`);
    return jsonError("Received an unreadable response from the image service.", 502);
  }

  // Cloudflare's /ai/run endpoint signals failure either via a non-2xx
  // HTTP status or via `success: false` in an otherwise 200 response —
  // check both rather than assuming only one applies.
  if (!cfRes.ok || data.success === false) {
    return cloudflareErrorResponse(cfRes, data, modelUsed, [apiToken, accountId]);
  }

  const base64 = data && data.result && data.result.image;

  if (!base64 || typeof base64 !== "string") {
    console.error(`Cloudflare response had no image data. model=${modelUsed} body=${JSON.stringify(data).slice(0, 500)}`);
    return jsonError(
      isEdit
        ? "The AI didn't return an edited image. Please try rephrasing your instruction."
        : "The AI didn't return an image. Please try rephrasing your prompt.",
      502
    );
  }

  // Both models used here (SDXL for generation, flux-2-klein-4b for
  // editing) document their output as a base64-encoded JPEG.
  return new Response(JSON.stringify({ imageDataUrl: `data:image/jpeg;base64,${base64}` }), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  });
}
