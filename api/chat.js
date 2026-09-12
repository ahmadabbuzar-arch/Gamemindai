// api/chat.js
// Vercel serverless function — secure proxy to the Groq API.
// The API key is read only from process.env.GROQ_API_KEY and is
// never sent to, or exposed in, the frontend.

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";

// Change this to swap models without touching any other code.
const GROQ_MODEL = process.env.GROQ_MODEL || "openai/gpt-oss-120b";

const UPSTREAM_TIMEOUT_MS = 25000;
const MAX_MESSAGES = 40;
const MAX_MESSAGE_LENGTH = 6000;

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed. Use POST." });
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    console.error("GROQ_API_KEY is not set.");
    return res.status(500).json({ error: "The server isn't configured yet. Please try again later." });
  }

  let body = req.body;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      return res.status(400).json({ error: "Invalid request body." });
    }
  }

  const { messages, system } = body || {};

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "No message provided." });
  }
  if (messages.length > MAX_MESSAGES) {
    return res.status(400).json({ error: "This conversation is too long. Please start a new chat." });
  }

  const cleanedMessages = [];
  for (const m of messages) {
    if (!m || typeof m.content !== "string" || !m.content.trim()) continue;
    const role = m.role === "assistant" ? "assistant" : "user";
    cleanedMessages.push({ role, content: m.content.slice(0, MAX_MESSAGE_LENGTH) });
  }

  if (cleanedMessages.length === 0) {
    return res.status(400).json({ error: "No valid message content provided." });
  }

  const systemPrompt =
    typeof system === "string" && system.trim()
      ? system.trim().slice(0, 2000)
      : "You are GameMind AI, a friendly, natural-sounding general-purpose AI assistant with strong gaming expertise. If the user writes in Hindi or Hinglish, reply in casual everyday spoken Hindi/Hinglish (like texting a friend), never shuddh/literary Hindi. Answer directly and briefly — usually 2 to 6 short paragraphs or a few bullet points, no long articles unless the user asks for detail. No emojis.";

  const payload = {
    model: GROQ_MODEL,
    messages: [{ role: "system", content: systemPrompt }, ...cleanedMessages],
    temperature: 0.8,
    max_tokens: 1024,
    stream: true,
  };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  try {
    const groqRes = await fetch(GROQ_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

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
      return res.status(502).json({ error: "GameMind AI couldn't get a response right now. Please try again." });
    }

    if (!groqRes.body || typeof groqRes.body.getReader !== "function") {
      clearTimeout(timeoutId);
      return res.status(502).json({ error: "GameMind AI couldn't get a response right now. Please try again." });
    }

    // From here on we stream plain text tokens straight through to the
    // browser as they arrive from Groq, so the reply appears the way it's
    // being written instead of popping in all at once.
    res.writeHead(200, {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    });

    const reader = groqRes.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let receivedAny = false;

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
            receivedAny = true;
            res.write(delta);
          }
        } catch {
          /* ignore a malformed SSE chunk and keep reading */
        }
      }
    }

    clearTimeout(timeoutId);

    if (!receivedAny) {
      // Headers are already sent as 200 plain text at this point, so we
      // can't switch to a JSON error response — the client treats an
      // empty stream as "no response" and shows its own error message.
      console.error("Groq stream completed with no content.");
    }

    return res.end();
  } catch (err) {
    clearTimeout(timeoutId);
    if (res.headersSent) {
      // We were mid-stream when this failed; just end the connection.
      // The client sees an incomplete/empty reply and surfaces its own
      // "didn't get a response" error rather than a broken partial one.
      console.error("Error while streaming Groq response:", err);
      try {
        return res.end();
      } catch {
        return;
      }
    }
    if (err.name === "AbortError") {
      return res.status(504).json({ error: "The AI service took too long to respond. Please try again." });
    }
    console.error("Unexpected error calling Groq API:", err);
    return res.status(500).json({ error: "Something went wrong on our end. Please try again." });
  }
};
