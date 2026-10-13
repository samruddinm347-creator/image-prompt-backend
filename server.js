'use strict';
// Promptly backend. No dependencies; needs Node 18+ (built-in fetch).
const http = require('http');
const fs = require('fs');
const path = require('path');

// Minimal .env loader so the key never has to live in code.
try {
  for (const line of fs.readFileSync(path.join(__dirname, '.env'), 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^(['"])(.*)\1$/, '$2');
  }
} catch { /* no .env file: rely on real environment variables */ }

const PORT = Number(process.env.PORT) || 3000;
const MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const API_BASE = (process.env.GEMINI_API_BASE || 'https://generativelanguage.googleapis.com').replace(/\/$/, '');
const MAX_IMAGE = 10 * 1024 * 1024;                       // matches the client limit
const MAX_BODY = Math.ceil(MAX_IMAGE * 4 / 3) + 64 * 1024; // base64 overhead + JSON
const MAX_NOTES = 500;

const SYSTEM = `You write prompts for AI image generators. You receive a reference image, a mode, and optional user instructions.
Study the image closely before writing: main subject and secondary objects, composition and framing, pose and positioning, camera perspective and lens feel, background and environment, lighting, shadows and colors, artistic style or photographic aesthetic, texture, atmosphere and mood, and other distinctive visible details. Mention text or logos only if clearly legible.
Then write ONE detailed, ready-to-use image-generation prompt that follows the mode.
Rules:
- Describe only what is visible. If something cannot be determined, stay general rather than inventing precise details (no camera models, brands, ages or locations you cannot see).
- Never name or identify real people; describe their appearance instead.
- Never claim to know the original prompt or generation settings of the reference.
- Explicit user instructions override the mode's default behavior.
- Output only the prompt as plain text in one to three paragraphs (about 120-250 words). No title, preamble, markdown or commentary.`;

const MODES = {
  recreate: ['Recreate Reference', 'Preserve as many recognizable visual details as possible: composition, framing, subject placement, pose, lighting, color palette, style and mood. The prompt should let a generator reproduce the image as closely as possible.'],
  changes: ['Make Changes', 'Keep every detail of the reference consistent except what the user asks to change. Apply only the requested changes, and explicitly describe the unaffected elements (subject, pose, composition, lighting, style) so they stay the same.'],
  scene: ['Change Scene', 'Keep the subject, artistic style, visual treatment, color grading and mood of the reference, but place it in a different setting. Use the scene the user specifies, or choose a fitting one, and adapt the lighting so it matches the new environment.'],
  concept: ['New Concept', 'Use the reference only as inspiration (its style, mood or a motif). You may change the concept, subject, composition and environment substantially to create a fresh, original idea.'],
  viral: ['Viral Trend', 'Capture the reference\'s visible aesthetic and trend style (for example retro film photography, a 90s look, a fashion portrait or another recognizable social-media look). Build the prompt around its distinctive characteristics: color grading, grain, lighting, framing, styling, props and mood. Name the trend only if it is clearly evident.']
};

function detectType(b) {
  if (b.length > 12 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return 'image/png';
  if (b.length > 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'image/jpeg';
  if (b.length > 12 && b.toString('latin1', 0, 4) === 'RIFF' && b.toString('latin1', 8, 12) === 'WEBP') return 'image/webp';
  return null;
}

function send(res, status, body, type = 'application/json; charset=utf-8') {
  res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
  res.end(type.startsWith('application/json') ? JSON.stringify(body) : body);
}
const fail = (res, status, error) => send(res, status, { error });

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let size = 0, over = false;
    req.on('data', c => {
      size += c.length;
      if (size > MAX_BODY) over = true;   // keep draining without buffering so the client gets a clean 413
      else if (!over) chunks.push(c);
    });
    req.on('end', () => over ? reject(Object.assign(new Error('too large'), { status: 413 })) : resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

// Pull the model's text out of an Interactions API response, skipping thought steps.
function extractText(d) {
  if (!d) return '';
  if (typeof d.output_text === 'string') return d.output_text.trim();
  const out = [];
  (function walk(n) {
    if (Array.isArray(n)) return n.forEach(walk);
    if (!n || typeof n !== 'object') return;
    if (n.type === 'thought' || n.type === 'thinking') return;
    if (n.type === 'text' && typeof n.text === 'string') out.push(n.text);
    else Object.values(n).forEach(walk);
  })(d.steps || d.outputs || []);
  return out.join('\n').trim();
}

function upstreamError(status, data) {
  const msg = String(data?.error?.message || '');
  if (status === 429) return [429, 'Gemini rate limit or quota reached. Wait a moment and try again, or check your plan.'];
  if (status === 401 || status === 403 || /api key/i.test(msg)) return [502, 'Gemini rejected the API key. Check GEMINI_API_KEY.'];
  if (status === 404) return [502, 'The configured Gemini model is not available. Set GEMINI_MODEL to a supported model.'];
  if (status >= 500) return [502, 'Gemini is temporarily unavailable. Try again shortly.'];
  return [502, 'Gemini could not process this request. Try a different image.'];
}

async function generate(req, res) {
  let payload;
  try { payload = JSON.parse(await readBody(req)); }
  catch (e) { return fail(res, e.status === 413 ? 413 : 400, e.status === 413 ? 'That image is too large. Use one under 10 MB.' : 'Invalid request.'); }

  const { image, mode } = payload || {};
  const notes = typeof payload?.instructions === 'string' ? payload.instructions.trim() : '';
  if (!MODES[mode]) return fail(res, 400, 'Choose a prompt mode.');
  if (notes.length > MAX_NOTES) return fail(res, 400, `Custom instructions are limited to ${MAX_NOTES} characters.`);
  if (mode === 'changes' && !notes) return fail(res, 400, 'Describe what to change in the custom instructions.');
  if (typeof image !== 'string' || !image) return fail(res, 400, 'Add a reference image first.');

  const bytes = Buffer.from(image, 'base64');
  if (bytes.length > MAX_IMAGE) return fail(res, 413, 'That image is over 10 MB. Choose a smaller file.');
  const mime = detectType(bytes);
  if (!mime) return fail(res, 415, 'Unsupported or invalid image. Use a PNG, JPG or WebP file.');

  const key = process.env.GEMINI_API_KEY;
  if (!key) return fail(res, 503, 'The server has no Gemini API key. Set GEMINI_API_KEY and restart.');

  const [label, directive] = MODES[mode];
  const text = `Mode: ${label}\nMode behavior: ${directive}\nUser instructions: ${notes || 'None'}\n\nStudy the reference image below and write the prompt.`;

  let r;
  try {
    r = await fetch(`${API_BASE}/v1beta/interactions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-goog-api-key': key },
      body: JSON.stringify({
        model: MODEL,
        system_instruction: SYSTEM,
        input: [{ type: 'text', text }, { type: 'image', data: bytes.toString('base64'), mime_type: mime }],
        generation_config: { temperature: 0.7 },
        store: false // ask Gemini not to retain this interaction
      }),
      signal: AbortSignal.timeout(90000)
    });
  } catch (e) {
    return e.name === 'TimeoutError'
      ? fail(res, 504, 'Gemini took too long to respond. Try again.')
      : fail(res, 502, 'Could not reach Gemini. Check the server\'s internet connection.');
  }

  const data = await r.json().catch(() => null);
  if (!r.ok) {
    console.error('Gemini error status', r.status);
    const [s, m] = upstreamError(r.status, data);
    return fail(res, s, m);
  }
  const prompt = extractText(data).replace(/^```\w*\n?|\n?```$/g, '').trim();
  if (!prompt) return fail(res, 502, 'The AI returned no prompt. Try again or use a different image.');
  send(res, 200, { prompt, mode });
}

const page = path.join(__dirname, 'index.html');
http.createServer(async (req, res) => {
  try {
    const url = req.url.split('?')[0];
    if (req.method === 'GET' && (url === '/' || url === '/index.html')) return send(res, 200, fs.readFileSync(page), 'text/html; charset=utf-8');
    if (req.method === 'POST' && url === '/api/generate') return await generate(req, res);
    fail(res, 404, 'Not found.');
  } catch (e) {
    console.error('Unhandled error:', e.name);
    if (!res.headersSent) fail(res, 500, 'Something went wrong on the server.');
  }
}).listen(PORT, () => {
  console.log(`Promptly running at http://localhost:${PORT}`);
  if (!process.env.GEMINI_API_KEY) console.warn('Warning: GEMINI_API_KEY is not set. Generation will fail until you add it.');
});
