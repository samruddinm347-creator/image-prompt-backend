'use strict';
// Promptly backend. No dependencies; needs Node 18+ (built-in fetch).
// Gemini call format verified against Google's docs (Interactions API, image understanding page).
const http = require('http');
const fs = require('fs');
const path = require('path');

// Minimal .env loader (real environment variables, e.g. Render's, always win).
try {
  for (const line of fs.readFileSync(path.join(__dirname, '.env'), 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^(['"])(.*)\1$/, '$2');
  }
} catch { /* no .env file */ }

const PORT = Number(process.env.PORT) || 3000;
const MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const API_BASE = (process.env.GEMINI_API_BASE || 'https://generativelanguage.googleapis.com').replace(/\/$/, '');
const MAX_IMAGE = 10 * 1024 * 1024;
const MAX_BODY = Math.ceil(MAX_IMAGE * 4 / 3) + 64 * 1024;
const MAX_NOTES = 500;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || '*'; // set to your website's address to lock it down
const ATTEMPTS = 3;                                        // Google says to retry 503 with backoff
const ATTEMPT_TIMEOUT_MS = 60000;
const RETRY_BASE_MS = Number(process.env.RETRY_BASE_MS) || 1500;

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

// ---------- safe logging (never the API key, never image data) ----------
function safeText(s, n = 300) {
  let t = String(s == null ? '' : s);
  const k = process.env.GEMINI_API_KEY;
  if (k) t = t.split(k).join('[key]');
  return t.replace(/AIza[\w-]{20,}/g, '[key]').replace(/[A-Za-z0-9+/=]{200,}/g, '[long-data]').replace(/\s+/g, ' ').slice(0, n);
}
function logError(where, e) {
  console.error(`${where}: ${e && e.name}: ${safeText(e && e.message)}${e && e.code ? ` (code ${e.code})` : ''}`);
  if (e && e.stack) console.error('  at ' + safeText(String(e.stack).split('\n').slice(1, 4).join(' | '), 400));
}
process.on('unhandledRejection', e => logError('Unhandled rejection', e));

// ---------- helpers ----------
const sleep = ms => new Promise(r => setTimeout(r, ms));

function detectType(b) {
  if (b.length > 12 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return 'image/png';
  if (b.length > 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'image/jpeg';
  if (b.length > 12 && b.toString('latin1', 0, 4) === 'RIFF' && b.toString('latin1', 8, 12) === 'WEBP') return 'image/webp';
  return null;
}

function send(res, status, body, type = 'application/json; charset=utf-8') {
  res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store', 'x-content-type-options': 'nosniff', 'access-control-allow-origin': ALLOWED_ORIGIN, 'vary': 'Origin' });
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

// Interactions API (new schema, default since May 2026): text is in steps[type=model_output].content[type=text].
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

// Error codes come from Google's "API errors" page.
const BLOCKED = new Set(['safety', 'recitation', 'language', 'prohibited_content', 'spii', 'blocklist', 'image_safety', 'image_prohibited_content', 'image_recitation', 'image_other', 'content_blocked']);
function upstreamError(status, code, msg) {
  if (BLOCKED.has(code)) return [422, 'Gemini blocked this image or request for safety or policy reasons. Try a different image or instructions.'];
  if (code === 'quota_exceeded') return [429, 'The Gemini daily quota is used up. Wait for it to reset, or check your plan and billing.'];
  if (status === 429) return [429, 'Gemini is getting too many requests. Wait a minute and try again.'];
  if (status === 402 || code === 'payment_required') return [402, 'The Gemini credit balance has run out. Add credits in Google AI Studio billing.'];
  if (status === 401 || code === 'authentication' || /api key/i.test(msg || '')) return [502, 'Gemini rejected the API key. Check GEMINI_API_KEY in your server settings.'];
  if (status === 403) return [502, 'This API key is not allowed to use Gemini. Check the key and its project settings in Google AI Studio.'];
  if (status === 404 || code === 'model_not_found') return [502, 'The configured Gemini model was not found. Set GEMINI_MODEL to a model listed in Google\'s docs.'];
  if (code === 'failed_precondition') return [502, 'The Google project is not ready for this request (for example billing is not enabled). Check Google AI Studio.'];
  if (status === 503) return [503, 'Gemini is overloaded right now. It was retried automatically. Please try again in a minute.'];
  if (status === 504) return [504, 'Gemini took too long to respond. Try again.'];
  if (status >= 500) return [502, 'Gemini had a temporary error. Please try again shortly.'];
  return [502, 'Gemini could not process this request. Try a different image. The server log has details.'];
}

// One Gemini call with automatic retry on 503 / 500 / per-minute 429 (Google: "wait and retry with exponential backoff").
async function callGemini(payload, key, meta) {
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    let r;
    try {
      r = await fetch(`${API_BASE}/v1beta/interactions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-goog-api-key': key },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS)
      });
    } catch (e) {
      logError(`Gemini request failed (attempt ${attempt}/${ATTEMPTS})`, e);
      return { netError: e };
    }
    const raw = await r.text().catch(() => '');
    let data = null; try { data = JSON.parse(raw); } catch { /* not JSON */ }
    if (r.ok) return { ok: true, data };

    const code = data && data.error && data.error.code, msg = data && data.error && data.error.message;
    console.error(`Gemini error: status=${r.status} code=${code || '-'} message="${safeText(msg || raw)}" model=${MODEL} mode=${meta.mode} image=${meta.mime} ${meta.kb}KB attempt=${attempt}/${ATTEMPTS}`);
    const retryable = r.status === 503 || r.status === 500 || (r.status === 429 && code !== 'quota_exceeded');
    if (!retryable || attempt === ATTEMPTS) return { status: r.status, code, msg };
    await sleep(RETRY_BASE_MS * attempt * attempt + Math.random() * 500);
  }
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
  if (!key) { console.error('GEMINI_API_KEY is not set'); return fail(res, 503, 'The server has no Gemini API key. Set GEMINI_API_KEY and redeploy.'); }

  const [label, directive] = MODES[mode];
  const text = `Mode: ${label}\nMode behavior: ${directive}\nUser instructions: ${notes || 'None'}\n\nStudy the reference image below and write the prompt.`;
  const result = await callGemini({
    model: MODEL,
    system_instruction: SYSTEM,
    input: [{ type: 'text', text }, { type: 'image', data: bytes.toString('base64'), mime_type: mime }],
    generation_config: { temperature: 0.7 },
    store: false // ask Gemini not to retain this interaction
  }, key, { mode, mime, kb: Math.round(bytes.length / 1024) });

  if (result.netError) {
    return result.netError.name === 'TimeoutError'
      ? fail(res, 504, 'Gemini took too long to respond. Try again.')
      : fail(res, 502, 'The server could not reach Gemini. Try again in a moment.');
  }
  if (!result.ok) { const [s, m] = upstreamError(result.status, result.code, result.msg); return fail(res, s, m); }

  const prompt = extractText(result.data).replace(/^```\w*\n?|\n?```$/g, '').trim();
  if (!prompt) {
    const types = ((result.data && (result.data.steps || result.data.outputs)) || []).map(s => s && s.type).join(',');
    console.error(`Gemini returned no text: interaction status=${result.data && result.data.status} step types=[${types}]`);
    return fail(res, 502, 'The AI returned no prompt. Try again or use a different image.');
  }
  send(res, 200, { prompt, mode });
}

// index.html lookup tolerates a different capitalization such as "Index.html" (Linux file names are case-sensitive).
function loadPage() {
  const exact = path.join(__dirname, 'index.html');
  if (fs.existsSync(exact)) return fs.readFileSync(exact);
  const alt = fs.readdirSync(__dirname).find(f => f.toLowerCase() === 'index.html');
  return alt ? fs.readFileSync(path.join(__dirname, alt)) : null;
}

http.createServer(async (req, res) => {
  try {
    const url = req.url.split('?')[0];
    if (req.method === 'OPTIONS') {
      res.writeHead(204, { 'access-control-allow-origin': ALLOWED_ORIGIN, 'access-control-allow-methods': 'POST, OPTIONS', 'access-control-allow-headers': 'content-type', 'access-control-max-age': '86400' });
      return res.end();
    }
    if (req.method === 'GET' && url === '/health') return send(res, 200, { ok: true, model: MODEL, keyConfigured: !!process.env.GEMINI_API_KEY });
    if (req.method === 'GET' && (url === '/' || url === '/index.html')) {
      const html = loadPage();
      if (!html) {
        console.error(`index.html not found next to server.js. Files here: ${fs.readdirSync(__dirname).join(', ')}`);
        return fail(res, 404, 'The website page (index.html) was not found on the server.');
      }
      return send(res, 200, html, 'text/html; charset=utf-8');
    }
    if (req.method === 'POST' && url === '/api/generate') return await generate(req, res);
    fail(res, 404, 'Not found.');
  } catch (e) {
    logError(`Unhandled error on ${req.method} ${String(req.url).split('?')[0]}`, e);
    if (!res.headersSent) fail(res, 500, 'Something went wrong on the server.');
  }
}).listen(PORT, () => {
  console.log(`Promptly running on port ${PORT} (model: ${MODEL}, API key ${process.env.GEMINI_API_KEY ? 'set' : 'NOT set'})`);
});
