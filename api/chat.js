// api/chat.js
// Vercel serverless function — secure proxy to the Groq API.
// The API key is read only from process.env.GROQ_API_KEY and is
// never sent to, or exposed in, the frontend.

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";

// Change this to swap models without touching any other code.
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

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
      : "You are GameMind AI, a friendly and knowledgeable assistant for gamers. Keep answers clear and practical. No emojis.";

  const payload = {
    model: GROQ_MODEL,
    messages: [{ role: "system", content: systemPrompt }, ...cleanedMessages],
    temperature: 0.8,
    max_tokens: 1024,
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

    clearTimeout(timeoutId);

    let data;
    try {
      data = await groqRes.json();
    } catch {
      return res.status(502).json({ error: "Received an unreadable response from the AI service." });
    }

    if (!groqRes.ok) {
      const upstreamMessage =
        data && data.error && data.error.message ? data.error.message : "The AI service returned an error.";
      console.error("Groq API error:", groqRes.status, upstreamMessage);
      return res.status(502).json({ error: "GameMind AI couldn't get a response right now. Please try again." });
    }

    const reply = data && data.choices && data.choices[0] && data.choices[0].message
      ? data.choices[0].message.content
      : "";

    if (!reply || !reply.trim()) {
      return res.status(502).json({ error: "GameMind AI didn't return a response. Please try again." });
    }

    return res.status(200).json({ reply: reply.trim() });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      return res.status(504).json({ error: "The AI service took too long to respond. Please try again." });
    }
    console.error("Unexpected error calling Groq API:", err);
    return res.status(500).json({ error: "Something went wrong on our end. Please try again." });
  }
};
