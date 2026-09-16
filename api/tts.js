// api/tts.js
// Vercel Function (Web/Fetch API handler signature) — secure proxy to the
// EasyVoice text-to-speech API. The key is read only from
// process.env.EASYVOICE_API_KEY and is never sent to, or exposed in, the
// frontend. Plain REST call — no SDK/npm install required.

const EASYVOICE_API_URL = "https://easyvoice.ae/api/v1/audio/speech";
const EASYVOICE_MODEL = "kokoro-82m";
const EASYVOICE_VOICE = process.env.EASYVOICE_VOICE || "af_aoede";
// Kokoro's Hindi voices — used only when the reply is in Hindi/Hinglish, so
// pronunciation doesn't sound like an English voice sounding out Hindi words.
const EASYVOICE_VOICE_HI = process.env.EASYVOICE_VOICE_HI || "hf_alpha";

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
// Fast, cheap model used only to transliterate romanized Hindi/Hinglish
// into Devanagari before sending it to the Hindi voice — Kokoro's Hindi
// phonemizer expects Devanagari script, not Roman letters.
const GROQ_TRANSLITERATE_MODEL = "openai/gpt-oss-20b";

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

// Very common Hinglish (Roman-script Hindi) words — used only to decide
// whether text needs the Hindi voice + Devanagari transliteration.
const HINGLISH_MARKERS = new Set([
  "hai", "hain", "nahi", "nahin", "kya", "kaise", "kaisa", "kaisi", "kyun", "kyu",
  "aap", "tum", "tumhe", "aapko", "mujhe", "hum", "humein", "unko", "unka",
  "mein", "main", "hoga", "hogi", "raha", "rahi", "rahe", "karo", "kare", "karna",
  "wala", "wali", "bhi", "abhi", "acha", "accha", "theek", "thik", "sahi",
  "bata", "batao", "dekho", "suno", "chalo", "matlab", "bahut", "bohot", "kuch",
]);

function looksHinglishOrHindi(text) {
  if (/[\u0900-\u097F]/.test(text)) return true; // already Devanagari
  const words = text.toLowerCase().match(/[a-z]+/g) || [];
  if (words.length === 0) return false;
  const hits = words.filter((w) => HINGLISH_MARKERS.has(w)).length;
  return hits / words.length > 0.12;
}

async function transliterateToDevanagari(text, groqApiKey) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 12000);
  try {
    const res = await fetch(GROQ_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${groqApiKey}`,
      },
      body: JSON.stringify({
        model: GROQ_TRANSLITERATE_MODEL,
        messages: [
          {
            role: "system",
            content:
              "Transliterate the ENTIRE given text into Devanagari script for text-to-speech — do not leave any words in Roman/Latin letters. Convert Hindi/Hinglish words (however they're spelled) into correct Devanagari, and also spell out any English words, brand names, or technical terms phonetically in Devanagari the way a Hindi speaker would naturally pronounce them while speaking Hindi (e.g. website → वेबसाइट, email → ईमेल). Numbers can stay as digits. The whole output must be one consistent script with no Roman-letter islands, so it reads smoothly instead of stopping at every script change. Output ONLY the transliterated text, nothing else — no explanation, no quotes.",
          },
          { role: "user", content: text },
        ],
        temperature: 0.2,
        max_tokens: 1500,
        stream: false,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!res.ok) return text; // fall back to the original text on any failure
    const data = await res.json();
    const out = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
    return typeof out === "string" && out.trim() ? out.trim() : text;
  } catch {
    clearTimeout(timeoutId);
    return text; // never block voice generation on this step failing
  }
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
  let cleanText = text.trim().slice(0, MAX_TEXT_LENGTH);

  // If the reply is in Hindi/Hinglish, transliterate to Devanagari and use
  // a Hindi voice so pronunciation is natural instead of an English voice
  // sounding out Hindi words letter-by-letter.
  let voice = EASYVOICE_VOICE;
  const groqApiKey = process.env.GROQ_API_KEY;
  if (looksHinglishOrHindi(cleanText) && groqApiKey) {
    cleanText = await transliterateToDevanagari(cleanText, groqApiKey);
    voice = EASYVOICE_VOICE_HI;
  }

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
        voice,
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
