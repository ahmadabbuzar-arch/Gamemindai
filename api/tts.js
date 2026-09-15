// api/tts.js
// Vercel Function (Web/Fetch API handler signature) — secure proxy to the
// EasyVoice text-to-speech API. The key is read only from
// process.env.EASYVOICE_API_KEY and is never sent to, or exposed in, the
// frontend. Plain REST call — no SDK/npm install required.

const EASYVOICE_API_URL = "https://easyvoice.ae/api/v1/audio/speech";
const EASYVOICE_MODEL = "kokoro-82m";
const EASYVOICE_VOICE = process.env.EASYVOICE_VOICE || "af_aoede";

const UPSTREAM_TIMEOUT_MS = 30000;
const MAX_TEXT_LENGTH = 4500; // stays safely under EasyVoice's free-tier per-request/day limits

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
  const apiKey = process.env.EASYVOICE_API_KEY;
  if (!apiKey) {
    console.error("EASYVOICE_API_KEY is not set.");
    return jsonError("Voice replies aren't configured yet. Please try again later.", 500);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return jsonError("Invalid request body.", 400);
  }

  const { text } = body || {};
  if (typeof text !== "string" || !text.trim()) {
    return jsonError("No text provided to speak.", 400);
  }
  const cleanText = text.trim().slice(0, MAX_TEXT_LENGTH);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let evRes;
  try {
    evRes = await fetch(EASYVOICE_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: EASYVOICE_MODEL,
        input: cleanText,
        voice: EASYVOICE_VOICE,
        response_format: "mp3",
      }),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      return jsonError("Voice generation took too long. Please try again.", 504);
    }
    console.error("Unexpected error calling EasyVoice API:", err);
    return jsonError("Couldn't reach the voice service. Check your connection and try again.", 500);
  }

  clearTimeout(timeoutId);

  if (!evRes.ok) {
    let errData = null;
    try {
      errData = await evRes.json();
    } catch {
      /* upstream error body wasn't JSON */
    }
    const status = evRes.status;
    console.error("EasyVoice API error:", status, JSON.stringify(errData).slice(0, 300));

    if (status === 401 || status === 403) {
      return jsonError("The voice service API key was rejected. Please check the server configuration.", 500);
    }
    if (status === 429) {
      return jsonError("Voice generation rate limit reached. Please wait a moment and try again.", 429);
    }
    if (status === 400) {
      return jsonError("That reply was too long or unsupported for voice. Please try a shorter message.", 400);
    }
    return jsonError("The voice service returned an error. Please try again.", 502);
  }

  let audioBuffer;
  try {
    audioBuffer = await evRes.arrayBuffer();
  } catch {
    return jsonError("Received an unreadable response from the voice service.", 502);
  }

  if (!audioBuffer || audioBuffer.byteLength === 0) {
    return jsonError("The voice service didn't return any audio. Please try again.", 502);
  }

  const base64 = Buffer.from(audioBuffer).toString("base64");
  const contentType = evRes.headers.get("content-type") || "audio/mpeg";

  return new Response(JSON.stringify({ audioDataUrl: `data:${contentType};base64,${base64}` }), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  });
}
