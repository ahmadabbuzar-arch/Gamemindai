/* =========================================================
   GameMind AI — /api/chat
   Vercel serverless function (Node.js runtime).

   - Reads GROQ_API_KEY only on the server (process.env).
   - Never expose the key to the client.
   - Builds a tool-specific system prompt, then calls Groq's
     OpenAI-compatible chat completions endpoint.
   ========================================================= */

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const DEFAULT_MODEL = "llama-3.3-70b-versatile";

const VALID_TOOLS = [
  "chat",
  "minecraft-helper",
  "story-generator",
  "quest-maker",
  "name-generator",
  "content-maker",
  "game-guide"
];

const VALID_GAMES = ["minecraft", "free-fire", "bgmi", "roblox", "gta-v", "other", null, undefined];

const GAME_LABELS = {
  minecraft: "Minecraft",
  "free-fire": "Free Fire",
  bgmi: "BGMI",
  roblox: "Roblox",
  "gta-v": "GTA V",
  other: "an unspecified game"
};

/* ---------------------------------------------------------
   Very small in-memory rate limiter (best-effort).
   Resets on cold start; fine for an MVP without a database.
   --------------------------------------------------------- */
const RATE_LIMIT_WINDOW_MS = 60 * 1000;
const RATE_LIMIT_MAX_REQUESTS = 20;
const rateLimitStore = new Map();

function isRateLimited(key) {
  const now = Date.now();
  const entry = rateLimitStore.get(key);
  if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW_MS) {
    rateLimitStore.set(key, { windowStart: now, count: 1 });
    return false;
  }
  entry.count += 1;
  if (entry.count > RATE_LIMIT_MAX_REQUESTS) {
    return true;
  }
  return false;
}

/* ---------------------------------------------------------
   System prompt builder
   --------------------------------------------------------- */
function gameLabel(game) {
  return GAME_LABELS[game] || "an unspecified game";
}

function baseGuidelines() {
  return (
    "You are GameMind AI, a friendly, upbeat assistant built for video game players. " +
    "Keep replies clear and conversational, formatted in plain text (no markdown headers or code fences unless the user is asking for actual code or commands). " +
    "Use short paragraphs or simple line breaks instead of dense walls of text. " +
    "You are not affiliated with or endorsed by any game publisher or studio. " +
    "When you are not fully certain about a specific game mechanic, version number, item stat, or in-game event, say so plainly instead of presenting a guess as confirmed fact. " +
    "Never claim a made-up detail is verified or official."
  );
}

function buildSystemPrompt(tool, game, options) {
  const g = gameLabel(game);
  const guidelines = baseGuidelines();

  switch (tool) {
    case "chat":
      return (
        guidelines +
        " You are in general chat mode. The player has not picked a specific game, so pay attention to whichever game they mention in their message (Minecraft, Free Fire, BGMI, Roblox, GTA V, or anything else) and adapt to it. " +
        "If it is unclear which game they mean, answer generally or ask one short clarifying question. Be helpful, friendly, and knowledgeable about gaming topics."
      );

    case "minecraft-helper":
      return (
        guidelines +
        " You are running the 'Minecraft Helper' tool. Its default focus is Minecraft" +
        (game && game !== "minecraft" ? ", but the player selected " + g + ", so adapt the same help style to that game instead" : "") +
        ". " +
        "The player picked the category: " +
        (options.category || "General") +
        ". Give practical, accurate help (commands, crafting recipes, mob behavior, build ideas, seeds, or addons) appropriate to that category. " +
        "If asked for a command, give the exact Java Edition or Bedrock Edition syntax and note which edition it applies to when it matters."
      );

    case "story-generator":
      return (
        guidelines +
        " You are running the 'Story Generator' tool. Write an original short story (roughly 200-400 words) inspired by " +
        g +
        ", in the style: " +
        (options.storyType || "Adventure") +
        ". The story should be original fiction, not a retelling of any copyrighted game plot, cutscene, or dialogue. Do not reproduce any song lyrics or copyrighted text."
      );

    case "quest-maker":
      return (
        guidelines +
        " You are running the 'Quest Maker' tool. Design an original custom quest or mission for " +
        g +
        " at difficulty: " +
        (options.difficulty || "Normal") +
        ". Include a short title, objective, 3-6 steps, and a suggested reward. Keep it usable as a homemade challenge, not an official in-game quest."
      );

    case "name-generator":
      return (
        guidelines +
        " You are running the 'Gamer Name Generator' tool. Generate 10 original username or character name ideas for " +
        g +
        " in a '" +
        (options.style || "Cool") +
        "' style. " +
        (options.keyword
          ? "Try to incorporate or riff on this keyword where it fits naturally: " + options.keyword + ". "
          : "") +
        "Return them as a simple numbered list, one name per line, with no extra commentary."
      );

    case "content-maker":
      return (
        guidelines +
        " You are running the 'Gaming Content Maker' tool for " +
        g +
        ". The creator wants a '" +
        (options.contentType || "Title") +
        "' for the platform: " +
        (options.platform || "YouTube") +
        ". If contentType is Title, give 5 punchy alternatives. If Description, write one ready-to-use description with a soft call to action. If Tags, give 15-20 comma-separated relevant tags. If Short Script, write a 30-60 second spoken script with brief scene notes. Keep everything original."
      );

    case "game-guide":
      return (
        guidelines +
        " You are running the 'Game Guide' tool for " +
        g +
        ". Answer the player's question with clear, structured, practical guidance. " +
        "If the question depends on a specific patch, season, or version you cannot verify, say your info may be out of date and suggest checking current in-game details."
      );

    default:
      return guidelines;
  }
}

/* ---------------------------------------------------------
   Validation
   --------------------------------------------------------- */
function validateBody(body) {
  if (!body || typeof body !== "object") {
    return "Request body must be a JSON object.";
  }
  const { tool, game, prompt, history } = body;

  if (!tool || VALID_TOOLS.indexOf(tool) === -1) {
    return "Missing or invalid 'tool'.";
  }
  if (game !== undefined && VALID_GAMES.indexOf(game) === -1) {
    return "Invalid 'game' value.";
  }
  if (tool !== "name-generator") {
    if (typeof prompt !== "string" || prompt.trim().length === 0) {
      return "Missing 'prompt'. Please enter some text.";
    }
  }
  if (typeof prompt === "string" && prompt.length > 4000) {
    return "Prompt is too long. Please shorten it (max 4000 characters).";
  }
  if (history !== undefined) {
    if (!Array.isArray(history) || history.length > 20) {
      return "Invalid conversation history.";
    }
    for (const turn of history) {
      if (
        !turn ||
        (turn.role !== "user" && turn.role !== "assistant") ||
        typeof turn.content !== "string" ||
        turn.content.length > 4000
      ) {
        return "Invalid conversation history entry.";
      }
    }
  }
  return null;
}

/* ---------------------------------------------------------
   Handler
   --------------------------------------------------------- */
module.exports = async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "GET") {
    // Lightweight health check the frontend uses on the Settings screen.
    return res.status(200).json({ configured: Boolean(process.env.GROQ_API_KEY) });
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "Method not allowed. Use POST." });
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return res.status(503).json({
      error:
        "GameMind AI's backend is not configured yet. Ask the site owner to add GROQ_API_KEY in Vercel's Environment Variables."
    });
  }

  let body = req.body;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch (e) {
      return res.status(400).json({ error: "Invalid JSON body." });
    }
  }

  const validationError = validateBody(body);
  if (validationError) {
    return res.status(400).json({ error: validationError });
  }

  const ipKey =
    (req.headers["x-forwarded-for"] || "").split(",")[0].trim() ||
    req.socket?.remoteAddress ||
    "unknown";
  if (isRateLimited(ipKey)) {
    return res.status(429).json({
      error: "You're sending requests a little too fast. Please wait a few seconds and try again."
    });
  }

  const { tool, game, prompt, options } = body;
  const history = Array.isArray(body.history) ? body.history : [];

  const systemPrompt = buildSystemPrompt(tool, game, options || {});

  const messages = [{ role: "system", content: systemPrompt }];
  history.forEach(function (turn) {
    messages.push({ role: turn.role, content: turn.content });
  });
  messages.push({ role: "user", content: prompt || "Give me some ideas." });

  const model = process.env.GROQ_MODEL || DEFAULT_MODEL;

  let groqResponse;
  try {
    groqResponse = await fetch(GROQ_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + apiKey
      },
      body: JSON.stringify({
        model: model,
        messages: messages,
        temperature: 0.7,
        max_tokens: 900
      })
    });
  } catch (networkErr) {
    return res.status(502).json({
      error: "Couldn't reach the AI service. Please check your connection and try again."
    });
  }

  if (!groqResponse.ok) {
    let details = "";
    try {
      const errJson = await groqResponse.json();
      details = (errJson && errJson.error && errJson.error.message) || "";
    } catch (e) {
      /* ignore parse failure */
    }

    if (groqResponse.status === 401) {
      return res.status(500).json({
        error: "The AI service rejected the configured API key. Please check GROQ_API_KEY in Vercel."
      });
    }
    if (groqResponse.status === 429) {
      return res.status(429).json({
        error: "The AI service is rate-limited right now. Please wait a moment and try again."
      });
    }
    return res.status(502).json({
      error: details
        ? "The AI service returned an error: " + details
        : "The AI service returned an error. Please try again."
    });
  }

  let data;
  try {
    data = await groqResponse.json();
  } catch (e) {
    return res.status(502).json({ error: "The AI service sent an unreadable response." });
  }

  const reply =
    data &&
    data.choices &&
    data.choices[0] &&
    data.choices[0].message &&
    data.choices[0].message.content;

  if (!reply) {
    return res.status(502).json({ error: "The AI service returned an empty response. Please try again." });
  }

  return res.status(200).json({ reply: reply.trim() });
};
