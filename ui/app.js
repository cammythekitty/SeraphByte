'use strict';
/* SeraphByte UI. Plain JS, no build step. Runs inside the desktop app's
 * webview (src/app.rs), which sets window.__SERAPH_WS__ to the engine's address.
 *
 * Engine protocol (src/server.rs):
 *   server → {"type":"models","models":[...],"dir","loaded"} on connect
 *   server → {"type":"model_info","model","arch","device","ctx_limit"} when a model is loaded
 *   client → {"type":"load","model","cuda"}   server → {"type":"log","line"}… then model_info or error
 *   client → {prompt, temperature, top_p, max_tokens, system_prompt}
 *   server → raw token text frames (reasoning as {"type":"think","text"}),
 *            then {"type":"done","tokens","prompt_tokens"}
 *            or {"type":"error","message"}
 * Each prompt is stateless (system + one user turn). Sessions here are display-only.
 */

// ── Config ──────────────────────────────────────────────────────────────────
const params   = new URLSearchParams(location.search);
const WS_URL   = window.__SERAPH_WS__ || params.get('ws') || 'ws://127.0.0.1:8543';
const CTX_MAX_FALLBACK = 8192;
const IDLE_FINALIZE_MS = 2500;
const CONTROL_TYPES = new Set(['models', 'model_info', 'log', 'think', 'done', 'error']);

const KEY_SESSIONS = 'seraph_sessions_v1';
const KEY_SETTINGS = 'seraph_settings_v1';
const KEY_MEMORY   = 'seraph_memory_v1';

// ── Elements ────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = {
  chat: document.querySelector('.chat'),
  log: $('log'), input: $('input'), composer: $('composer'),
  send: $('send-btn'), stop: $('stop-btn'),
  newChat: $('new-chat'), exportBtn: $('export-btn'), clearBtn: $('clear-btn'),
  sessions: $('sessions'),
  convLabel: $('conv-label'), convTitle: $('conv-title'), convSub: $('conv-sub'),
  paramLine: $('param-line'),
  modelList: $('model-list'), modelStatus: $('model-status'), deviceToggle: $('device-toggle'),
  emptyNote: document.querySelector('#empty .mono'),
  status: $('status'), statusText: $('status-text'),
  tps: $('tps'), ttft: $('ttft'),
  ctxVal: $('ctx-val'), ctxFill: $('ctx-fill'), ctxNote: $('ctx-note'),
  devVal: $('dev-val'), archVal: $('arch-val'),
  temp: $('temp'), topp: $('topp'), maxtok: $('maxtok'),
  tempVal: $('temp-val'), toppVal: $('topp-val'), maxtokVal: $('maxtok-val'),
  sys: $('sysprompt'), pins: $('pins'),
};

// ── Storage helpers ─────────────────────────────────────────────────────────
function loadJSON(key, fallback) {
  try { const v = JSON.parse(localStorage.getItem(key)); return v ?? fallback; }
  catch { return fallback; }
}
function saveJSON(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage full or blocked */ }
}

// ── Small utils ─────────────────────────────────────────────────────────────
function roman(n) {
  if (!n || n < 1) return '';
  const map = [[1000,'M'],[900,'CM'],[500,'D'],[400,'CD'],[100,'C'],[90,'XC'],[50,'L'],[40,'XL'],[10,'X'],[9,'IX'],[5,'V'],[4,'IV'],[1,'I']];
  let out = '';
  for (const [v, s] of map) while (n >= v) { out += s; n -= v; }
  return out;
}
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const estTokens = (s) => Math.ceil((s || '').length / 4);
const isToday = (ts) => new Date(ts).toDateString() === new Date().toDateString();

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; }
  catch {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    let ok = false; try { ok = document.execCommand('copy'); } catch {}
    ta.remove(); return ok;
  }
}
function flash(btn, label) {
  const orig = btn.dataset.label || btn.textContent;
  btn.dataset.label = orig;
  btn.textContent = label;
  setTimeout(() => { btn.textContent = orig; }, 1400);
}

// ── Settings ────────────────────────────────────────────────────────────────
const SAMPLING_DEFAULTS = { temperature: 0.7, top_p: 0.9, max_tokens: 512 };
const settings = Object.assign(
  { ...SAMPLING_DEFAULTS, system_prompt: '', lastModel: '', cuda: true },
  loadJSON(KEY_SETTINGS, {})
);
function applySettingsToUI() {
  el.temp.value = settings.temperature;
  el.topp.value = settings.top_p;
  el.maxtok.value = settings.max_tokens;
  el.sys.value = settings.system_prompt;
  syncSettingLabels();
}
function syncSettingLabels() {
  const t = (+settings.temperature).toFixed(2).replace(/0$/, '');
  const p = (+settings.top_p).toFixed(2).replace(/0$/, '');
  el.tempVal.textContent = t;
  el.toppVal.textContent = p;
  el.maxtokVal.textContent = settings.max_tokens;
  el.paramLine.textContent = `temp ${t} · top-p ${p} · max ${settings.max_tokens}`;
  // Very high temperature without a top-p cutoff samples near-uniformly from the
  // whole vocabulary, which reads as random multilingual tokens.
  const warn = $('sampling-warn');
  const t2 = +settings.temperature, p2 = +settings.top_p;
  let msg = '';
  if (t2 >= 1.5 && p2 >= 0.98) msg = 'Temperature this high with top-p at 1 gives near-random output. Lower one of them.';
  else if (t2 >= 1.3) msg = 'High temperature: expect output to drift or fall apart.';
  warn.textContent = msg;
  warn.hidden = !msg;
}
$('sampling-reset').addEventListener('click', () => {
  Object.assign(settings, SAMPLING_DEFAULTS);
  saveJSON(KEY_SETTINGS, settings);
  applySettingsToUI();
});
function bindSlider(input, key, parse) {
  input.addEventListener('input', () => {
    settings[key] = parse(input.value);
    syncSettingLabels();
    saveJSON(KEY_SETTINGS, settings);
  });
}
bindSlider(el.temp, 'temperature', parseFloat);
bindSlider(el.topp, 'top_p', parseFloat);
bindSlider(el.maxtok, 'max_tokens', (v) => parseInt(v, 10));
el.sys.addEventListener('input', () => {
  settings.system_prompt = el.sys.value;
  saveJSON(KEY_SETTINGS, settings);
});

// ── Pinned memory (kept compatible with the old app's key) ──────────────────
const Memory = {
  all() { const v = loadJSON(KEY_MEMORY, []); return Array.isArray(v) ? v : []; },
  add(text) {
    const items = this.all();
    items.push({ id: Date.now(), text: text.trim(), tag: 'pin', ts: new Date().toISOString() });
    saveJSON(KEY_MEMORY, items);
  },
  remove(id) { saveJSON(KEY_MEMORY, this.all().filter((e) => e.id !== id)); },
};
function renderPins() {
  const items = Memory.all();
  el.pins.replaceChildren();
  if (!items.length) {
    const li = document.createElement('li');
    li.className = 'none';
    li.textContent = 'Nothing pinned. Use Pin under a reply.';
    el.pins.appendChild(li);
    return;
  }
  for (const item of items) {
    const li = document.createElement('li');
    const span = document.createElement('span');
    span.className = 'pin-text';
    span.textContent = item.text;
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'btn-mini';
    rm.textContent = '×';
    rm.setAttribute('aria-label', 'Remove pinned item');
    rm.addEventListener('click', () => { Memory.remove(item.id); renderPins(); });
    li.append(span, rm);
    el.pins.appendChild(li);
  }
}

// ── Sessions (display transcripts, localStorage only) ───────────────────────
const store = (() => {
  const s = loadJSON(KEY_SESSIONS, null);
  if (s && Array.isArray(s.list)) return s;
  return { seq: 0, current: null, list: [] };
})();
function persistSessions() { saveJSON(KEY_SESSIONS, store); }
function newSession() {
  const now = Date.now();
  const s = { id: `s${now}`, seq: ++store.seq, title: '', created: now, updated: now, messages: [] };
  store.list.push(s);
  store.current = s.id;
  persistSessions();
  return s;
}
function currentSession() {
  return store.list.find((s) => s.id === store.current) || newSession();
}
function sessionTitle(s) { return s.title || 'New conversation'; }

function renderSessions() {
  const cur = currentSession();
  const visible = store.list
    .filter((s) => s.messages.length || s.id === cur.id)
    .sort((a, b) => b.updated - a.updated);
  const today = visible.filter((s) => isToday(s.updated));
  const earlier = visible.filter((s) => !isToday(s.updated));
  el.sessions.replaceChildren();
  const addGroup = (label, items) => {
    if (!items.length) return;
    const h = document.createElement('span');
    h.className = 'label group-label';
    h.textContent = label;
    el.sessions.appendChild(h);
    for (const s of items) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'session';
      b.dataset.id = s.id;
      if (s.id === cur.id) b.setAttribute('aria-current', 'true');
      const n = s.messages.length;
      const meta = (s.id === cur.id ? 'open · ' : '') + `${n} message${n === 1 ? '' : 's'}`;
      b.innerHTML = '<span class="num"></span><span class="body"><span class="title"></span><span class="meta"></span></span>';
      b.querySelector('.num').textContent = roman(s.seq) + '.';
      b.querySelector('.title').textContent = sessionTitle(s);
      b.querySelector('.meta').textContent = meta;
      el.sessions.appendChild(b);
    }
  };
  addGroup('Today', today);
  addGroup('Earlier', earlier);
  syncBusy();
}
el.sessions.addEventListener('click', (e) => {
  const b = e.target.closest('.session');
  if (!b || gen) return;
  store.current = b.dataset.id;
  persistSessions();
  renderAll();
});
el.newChat.addEventListener('click', () => {
  if (gen) return;
  if (currentSession().messages.length) newSession();
  renderAll();
  el.input.focus();
});
el.clearBtn.addEventListener('click', () => {
  const s = currentSession();
  if (gen || !s.messages.length) return;
  if (!confirm('Clear this conversation?')) return;
  s.messages = [];
  s.title = '';
  s.updated = Date.now();
  persistSessions();
  renderAll();
});
el.exportBtn.addEventListener('click', () => {
  const s = currentSession();
  if (!s.messages.length) return;
  const lines = [`# ${sessionTitle(s)}`, ''];
  for (const m of s.messages) {
    lines.push(m.role === 'user' ? '## You' : '## Seraph', '', m.text, '');
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `seraph-${(s.title || 'conversation').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48) || 'conversation'}.md`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ── Markdown ────────────────────────────────────────────────────────────────
let mdReady = false;
function initMarkdown() {
  if (mdReady || !window.marked || !window.DOMPurify) return;
  marked.use({ gfm: true, breaks: false });
  DOMPurify.addHook('afterSanitizeAttributes', (node) => {
    if (node.tagName === 'A' && node.getAttribute('href')) {
      node.setAttribute('target', '_blank');
      node.setAttribute('rel', 'noopener noreferrer');
    }
  });
  mdReady = true;
}
function patchUnclosedFence(text) {
  const fences = text.match(/^ {0,3}(```|~~~)/gm);
  return fences && fences.length % 2 ? text + '\n```' : text;
}
function renderMarkdown(target, text, streaming) {
  initMarkdown();
  const src = streaming ? patchUnclosedFence(text) : text;
  if (mdReady) {
    target.innerHTML = DOMPurify.sanitize(marked.parse(src));
  } else {
    target.textContent = src; // libs failed to load: still readable
  }
  decorateCode(target);
  if (streaming) {
    let host = target;
    while (host.lastElementChild && /^(P|UL|OL|LI|BLOCKQUOTE|H[1-6])$/.test(host.lastElementChild.tagName)) {
      host = host.lastElementChild;
    }
    const c = document.createElement('span');
    c.className = 'cursor';
    c.setAttribute('aria-hidden', 'true');
    host.appendChild(c);
  }
}
function decorateCode(root) {
  for (const pre of root.querySelectorAll('pre')) {
    if (pre.parentElement.classList.contains('codeblock')) continue;
    const code = pre.querySelector('code');
    const lang = (code && (code.className.match(/language-([\w+#.-]+)/) || [])[1]) || 'Code';
    const wrap = document.createElement('div');
    wrap.className = 'codeblock';
    const head = document.createElement('div');
    head.className = 'codeblock-head';
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = lang;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-mini code-copy';
    btn.textContent = 'Copy';
    head.append(label, btn);
    pre.replaceWith(wrap);
    wrap.append(head, pre);
  }
}

// Delegated actions: the streaming DOM is rebuilt every frame.
el.log.addEventListener('click', async (e) => {
  const codeBtn = e.target.closest('.code-copy');
  if (codeBtn) {
    const pre = codeBtn.closest('.codeblock').querySelector('pre');
    flash(codeBtn, (await copyText(pre.innerText)) ? 'Copied' : 'Failed');
    return;
  }
  const act = e.target.closest('[data-act]');
  if (!act) return;
  const msg = act.closest('.msg');
  const record = msg && msg._record;
  if (!record) return;
  if (act.dataset.act === 'copy') {
    flash(act, (await copyText(record.text)) ? 'Copied' : 'Failed');
  } else if (act.dataset.act === 'pin') {
    const t = record.text.replace(/\s+/g, ' ').trim();
    Memory.add(t.length > 160 ? t.slice(0, 160) + '…' : t);
    renderPins();
    flash(act, 'Pinned');
  }
});

// ── Chat rendering ──────────────────────────────────────────────────────────
const DIAMOND = '<svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 0l5 5-5 5-5-5z"/></svg>';

function scroller() {
  return matchMedia('(max-width: 820px)').matches ? document.scrollingElement : el.log;
}
function nearBottom() {
  const s = scroller();
  return s.scrollHeight - s.scrollTop - s.clientHeight < 96;
}
function scrollToBottom() { const s = scroller(); s.scrollTop = s.scrollHeight; }

function appendDividerIfNeeded() {
  if (!el.log.lastElementChild) return;
  const d = document.createElement('div');
  d.className = 'divider';
  d.setAttribute('aria-hidden', 'true');
  d.innerHTML = DIAMOND;
  el.log.appendChild(d);
}
function renderUser(record) {
  appendDividerIfNeeded();
  const m = document.createElement('div');
  m.className = 'msg user';
  m.innerHTML = '<span class="who">You</span><p class="text"></p>';
  m.querySelector('.text').textContent = record.text;
  el.log.appendChild(m);
  return m;
}
function renderAssistantShell() {
  appendDividerIfNeeded();
  const m = document.createElement('div');
  m.className = 'msg assistant';
  m.innerHTML = '<span class="who">Seraph</span><div class="md"></div>';
  el.log.appendChild(m);
  return m;
}
function statsLine(st) {
  if (!st) return '';
  const parts = [];
  if (st.tokens != null) parts.push(`${fmtInt(st.tokens)} token${st.tokens === 1 ? '' : 's'}`);
  if (st.tps != null) parts.push(`${st.tps.toFixed(1)} tok/s`);
  if (st.ttft != null) parts.push(`${st.ttft.toFixed(1)} s first token`);
  if (st.end === 'stopped') parts.push('stopped');
  if (st.end === 'disconnected') parts.push('connection lost');
  if (st.end === 'error') parts.push(`error: ${st.error || 'unknown'}`);
  return parts.join(' · ');
}
function renderAssistantFoot(m, record) {
  m.querySelector('.msg-foot')?.remove();
  const foot = document.createElement('div');
  foot.className = 'msg-foot';
  const stats = document.createElement('span');
  stats.className = 'stats';
  if (record.stats && (record.stats.end === 'error' || record.stats.end === 'disconnected')) stats.classList.add('warn');
  stats.textContent = statsLine(record.stats);
  foot.appendChild(stats);
  if (record.text) {
    for (const [act, label] of [['copy', 'Copy'], ['pin', 'Pin']]) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn-mini';
      b.dataset.act = act;
      b.textContent = label;
      foot.appendChild(b);
    }
  }
  m.appendChild(foot);
  m._record = record;
}
function thinkBlock(m) {
  let d = m.querySelector('.think');
  if (!d) {
    d = document.createElement('details');
    d.className = 'think';
    d.innerHTML = '<summary><span class="label">Reasoning</span></summary><div class="think-body"></div>';
    m.querySelector('.md').before(d);
  }
  return d;
}
function renderAssistantFinal(record) {
  const m = renderAssistantShell();
  if (record.think) thinkBlock(m).querySelector('.think-body').textContent = record.think;
  renderMarkdown(m.querySelector('.md'), record.text, false);
  renderAssistantFoot(m, record);
  return m;
}

function renderHeader() {
  const s = currentSession();
  el.convLabel.textContent = `Conversation ${roman(s.seq)}`;
  el.convTitle.textContent = sessionTitle(s);
  el.convSub.textContent = engine.model
    ? [engine.model, engine.device, engine.ctx ? `context ${fmtInt(engine.ctx)}` : null].filter(Boolean).join(' · ')
    : (engine.loading ? `loading ${engine.loading}` : 'no model loaded');
  document.title = s.title ? `${s.title} · SeraphByte` : 'SeraphByte';
}
function renderLog() {
  const s = currentSession();
  el.log.replaceChildren();
  for (const m of s.messages) {
    if (m.role === 'user') renderUser(m);
    else renderAssistantFinal(m);
  }
  el.chat.classList.toggle('is-empty', !s.messages.length);
  scrollToBottom();
  // Show the last reply's numbers in the ledger.
  const last = [...s.messages].reverse().find((m) => m.role === 'assistant' && m.stats);
  showStats(last ? last.stats : null);
}
function renderAll() {
  renderHeader();
  renderLog();
  renderSessions();
}

// ── Ledger ──────────────────────────────────────────────────────────────────
const engine = { model: null, arch: null, device: null, ctx: null, models: [], dir: '', loading: null, note: '', warn: false };

function ctxMax() { return engine.ctx || CTX_MAX_FALLBACK; }
function showContext(used, exact) {
  const max = ctxMax();
  const pct = Math.min(100, (used / max) * 100);
  el.ctxVal.textContent = `${exact || !used ? '' : '~'}${fmtInt(used)} / ${fmtInt(max)}`;
  el.ctxFill.style.width = `${pct}%`;
  el.ctxFill.classList.toggle('high', pct > 80);
  el.ctxNote.textContent = exact
    ? 'Last turn: system, prompt and reply, as counted by the engine.'
    : `Last turn only. Estimated at four characters per token${engine.ctx ? '' : ', against an assumed 8,192 limit'}.`;
}
function showStats(st) {
  el.tps.textContent = st && st.tps != null ? st.tps.toFixed(1) : '–';
  el.ttft.textContent = st && st.ttft != null ? st.ttft.toFixed(1) : '–';
  el.ttft.nextElementSibling.hidden = !(st && st.ttft != null);
  if (st && st.ctxUsed != null) showContext(st.ctxUsed, !!st.ctxExact);
  else showContext(0, false);
}
const DIAMOND_SMALL = '<svg class="diamond" width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 .7l4.3 4.3-4.3 4.3-4.3-4.3z"/></svg>';
const onCuda = (device) => /^CUDA/i.test(device || '');

function showEngine() {
  el.deviceToggle.textContent = settings.cuda ? 'Load on CUDA' : 'Load on CPU';
  el.modelList.replaceChildren();
  for (const name of engine.models) {
    const loaded = name === engine.model;
    const loading = name === engine.loading;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'model-entry' + (loaded ? ' loaded' : '') + (loading ? ' loading' : '');
    b.setAttribute('role', 'listitem');
    b.dataset.model = name;
    b.disabled = !!engine.loading || !!gen || !socketOpen();
    b.innerHTML = `<span class="model-name-row">${DIAMOND_SMALL}<span class="model-name"></span></span><span class="model-meta"></span>`;
    b.querySelector('.diamond').classList.toggle('on', loaded);
    b.querySelector('.model-name').textContent = name.replace(/\.gguf$/i, '');
    b.querySelector('.model-meta').textContent = loading
      ? 'loading…'
      : loaded
        ? [engine.arch, engine.device, engine.ctx ? `ctx ${fmtInt(engine.ctx)}` : null].filter(Boolean).join(' · ')
        : 'GGUF · not loaded';
    if (loaded) b.title = 'Loaded. Switch device above and click to reload on it.';
    el.modelList.appendChild(b);
  }
  let note = engine.note;
  if (!note) {
    if (!socketOpen()) note = 'waiting for engine';
    else if (!engine.models.length) note = engine.dir ? `No .gguf files in ${engine.dir}` : '';
    else if (!engine.model && !engine.loading) note = 'Choose a model to load.';
  }
  el.modelStatus.textContent = note;
  el.modelStatus.classList.toggle('warn', engine.warn);
  el.emptyNote.textContent = engine.model
    ? 'Each message is answered on its own. The model does not see earlier turns yet.'
    : engine.loading ? `Loading ${engine.loading}…` : 'Choose a model on the left to begin.';
  el.devVal.textContent = engine.device || '–';
  el.archVal.textContent = engine.arch || '–';
  renderHeader();
}

function loadModel(name) {
  if (!socketOpen() || engine.loading || gen) return;
  engine.loading = name;
  engine.note = '';
  engine.warn = false;
  socket.send(JSON.stringify({ type: 'load', model: name, cuda: settings.cuda }));
  showEngine();
  syncBusy();
}
el.modelList.addEventListener('click', (e) => {
  const b = e.target.closest('.model-entry');
  if (!b || b.disabled) return;
  const name = b.dataset.model;
  // Reload a loaded model only when the chosen device differs.
  if (name === engine.model && onCuda(engine.device) === settings.cuda) return;
  loadModel(name);
});
el.deviceToggle.addEventListener('click', () => {
  settings.cuda = !settings.cuda;
  saveJSON(KEY_SETTINGS, settings);
  showEngine();
});
function setStatus(text, cls) {
  el.statusText.textContent = text;
  el.status.className = 'status' + (cls ? ' ' + cls : '');
}

// ── Socket ──────────────────────────────────────────────────────────────────
let socket = null;
let reconnectTimer = null;
let backoff = 1500;
let doneSeen = false;   // engine sends {"type":"done"}; until seen, fall back to idle finalize

function socketOpen() { return socket && socket.readyState === WebSocket.OPEN; }

function connect(delay = 0) {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    setStatus('connecting');
    let ws;
    try { ws = new WebSocket(WS_URL); }
    catch { scheduleReconnect(); return; }
    socket = ws;
    ws.onopen = () => {
      if (socket !== ws) return;
      backoff = 1500;
      setStatus('connected', 'on');
      showEngine();
      syncBusy();
    };
    ws.onmessage = (ev) => { if (socket === ws) onFrame(ev.data); };
    ws.onclose = () => {
      if (socket !== ws) return;
      socket = null;
      if (gen) finalize('disconnected');
      setStatus('disconnected');
      if (engine.loading) { engine.loading = null; engine.note = 'Connection lost while loading.'; engine.warn = true; }
      showEngine();
      syncBusy();
      scheduleReconnect();
    };
    ws.onerror = () => { /* onclose follows */ };
  }, delay);
}
function scheduleReconnect() {
  connect(backoff);
  backoff = Math.min(backoff * 1.6, 10000);
}

function parseControl(raw) {
  if (typeof raw !== 'string' || raw.charCodeAt(0) !== 123 /* { */) return null;
  try {
    const v = JSON.parse(raw);
    if (v && typeof v === 'object' && !Array.isArray(v) && CONTROL_TYPES.has(v.type)) return v;
  } catch { /* plain token text */ }
  return null;
}

function onFrame(raw) {
  if (typeof raw !== 'string') return;
  const ctl = parseControl(raw);
  if (ctl) {
    if (ctl.type === 'models') {
      engine.models = Array.isArray(ctl.models) ? ctl.models.filter((m) => typeof m === 'string') : [];
      engine.dir = ctl.dir || '';
      if (!ctl.loaded) Object.assign(engine, { model: null, arch: null, device: null, ctx: null });
      showEngine();
      syncBusy();
      // Bring back the last model automatically (or the only one there is).
      if (!ctl.loaded && !engine.loading) {
        const pick = engine.models.includes(settings.lastModel) ? settings.lastModel
          : engine.models.length === 1 ? engine.models[0] : null;
        if (pick) loadModel(pick);
      }
    } else if (ctl.type === 'model_info') {
      engine.model = ctl.model || null;
      engine.arch = ctl.arch || null;
      engine.device = ctl.device || null;
      engine.ctx = Number(ctl.ctx_limit) || null;
      engine.loading = null;
      engine.note = '';
      engine.warn = false;
      if (engine.model) { settings.lastModel = engine.model; saveJSON(KEY_SETTINGS, settings); }
      showEngine();
      syncBusy();
      showContext(0, false);
    } else if (ctl.type === 'think') {
      if (typeof ctl.text === 'string') onToken(ctl.text, true);
    } else if (ctl.type === 'log') {
      if (engine.loading) { engine.note = String(ctl.line || ''); showEngine(); }
    } else if (ctl.type === 'done') {
      doneSeen = true;
      if (gen) finalize('done', ctl);
    } else if (ctl.type === 'error') {
      if (gen) finalize('error', ctl);
      else if (engine.loading) {
        engine.loading = null;
        engine.note = String(ctl.message || 'Load failed');
        engine.warn = true;
        showEngine();
        syncBusy();
      } else console.warn('Engine error:', ctl.message);
    }
    return;
  }
  onToken(raw);
}

// ── Generation ──────────────────────────────────────────────────────────────
let gen = null;       // active reply
let idleGen = null;   // reply closed by the idle fallback; reopened if more tokens arrive

function syncBusy() {
  const busy = !!gen;
  el.send.disabled = busy || !socketOpen() || !engine.model || !!engine.loading;
  el.stop.disabled = !busy;
  el.newChat.disabled = busy;
  el.clearBtn.disabled = busy;
  for (const b of el.sessions.querySelectorAll('.session')) b.disabled = busy && b.getAttribute('aria-current') !== 'true';
  el.log.setAttribute('aria-busy', busy ? 'true' : 'false');
  for (const b of el.modelList.querySelectorAll('.model-entry')) b.disabled = busy || !!engine.loading || !socketOpen();
  if (socketOpen()) {
    if (busy) setStatus('generating', 'on busy');
    else if (engine.loading) setStatus('loading model', 'on busy');
    else setStatus(engine.model ? 'ready' : 'connected', 'on');
  }
}

function send() {
  const text = el.input.value.trim();
  if (!text || gen || !socketOpen() || !engine.model || engine.loading) return;
  const s = currentSession();
  const userRec = { role: 'user', text, ts: Date.now() };
  s.messages.push(userRec);
  if (!s.title) { const t = text.replace(/\s+/g, ' '); s.title = t.length > 48 ? t.slice(0, 48).trimEnd() + '…' : t; }
  s.updated = Date.now();
  persistSessions();

  el.chat.classList.remove('is-empty');
  renderUser(userRec);
  const m = renderAssistantShell();
  const body = m.querySelector('.md');
  body.innerHTML = '<span class="waiting">Thinking<span class="cursor" aria-hidden="true"></span></span>';

  const sys = settings.system_prompt.trim();
  gen = {
    msg: m, body, text: '', think: '', frames: 0, thinkOpenedByUs: false,
    sessionId: s.id, prompt: text, system: sys,
    sentAt: performance.now(), firstAt: 0, lastAt: 0,
    raf: 0, idle: 0, record: null,
  };
  idleGen = null;
  socket.send(JSON.stringify({
    prompt: text,
    temperature: settings.temperature,
    top_p: settings.top_p,
    max_tokens: settings.max_tokens,
    system_prompt: sys || null,
  }));

  el.input.value = '';
  autosize();
  renderHeader();
  renderSessions();
  showContext(estTokens(sys) + estTokens(text), false);
  scrollToBottom();
}

function onToken(raw, think = false) {
  if (!gen && idleGen) {
    gen = idleGen; idleGen = null;
    gen.msg.querySelector('.msg-foot')?.remove();
    syncBusy();
  }
  if (!gen) return;
  const now = performance.now();
  if (!gen.firstAt) {
    gen.firstAt = now;
    el.ttft.textContent = ((now - gen.sentAt) / 1000).toFixed(1);
    el.ttft.nextElementSibling.hidden = false;
  }
  gen.lastAt = now;
  gen.frames++;
  if (think) gen.think += raw;
  else gen.text += raw;
  scheduleRender();
  if (!doneSeen) {
    clearTimeout(gen.idle);
    gen.idle = setTimeout(() => {
      if (!gen) return;
      const g = gen;
      finalize('idle');
      idleGen = g;
    }, IDLE_FINALIZE_MS);
  }
}

function scheduleRender() {
  if (gen.raf) return;
  gen.raf = requestAnimationFrame(() => {
    if (!gen) return;
    gen.raf = 0;
    const stick = nearBottom();
    if (gen.think) {
      const d = thinkBlock(gen.msg);
      const tb = d.querySelector('.think-body');
      tb.textContent = gen.think;
      // Open while only reasoning has arrived; fold it away once the answer starts.
      if (!gen.text && !gen.thinkOpenedByUs) { d.open = true; gen.thinkOpenedByUs = true; }
      if (gen.text && gen.thinkOpenedByUs) { d.open = false; gen.thinkOpenedByUs = false; }
      if (d.open) tb.scrollTop = tb.scrollHeight;
    }
    if (gen.text) renderMarkdown(gen.body, gen.text, true);
    else if (gen.think) gen.body.replaceChildren();
    const secs = (gen.lastAt - gen.firstAt) / 1000;
    if (gen.frames > 1 && secs > 0) el.tps.textContent = ((gen.frames - 1) / secs).toFixed(1);
    showContext(estTokens(gen.system) + estTokens(gen.prompt) + estTokens(gen.text), false);
    if (stick) scrollToBottom();
  });
}

function finalize(end, info) {
  const g = gen;
  if (!g) return;
  gen = null;
  cancelAnimationFrame(g.raf);
  g.raf = 0;
  clearTimeout(g.idle);

  const exact = end === 'done' && Number.isFinite(info?.tokens) && Number.isFinite(info?.prompt_tokens);
  const tokens = exact ? info.tokens : g.frames;
  const secs = g.firstAt ? (g.lastAt - g.firstAt) / 1000 : 0;
  // Rate over the interval between first and last frame, so the first token isn't counted.
  const tps = tokens > 1 && secs > 0.05 ? (tokens - 1) / secs : null;
  const stats = {
    end: end === 'idle' ? 'done' : end,
    tokens: g.firstAt ? tokens : 0,
    tps,
    ttft: g.firstAt ? (g.firstAt - g.sentAt) / 1000 : null,
    ctxUsed: exact ? info.prompt_tokens + info.tokens : estTokens(g.system) + estTokens(g.prompt) + estTokens(g.text),
    ctxExact: exact,
    error: end === 'error' ? String(info?.message || '') : undefined,
  };

  const s = store.list.find((x) => x.id === g.sessionId);
  if (g.record) {
    g.record.text = g.text;
    g.record.think = g.think || undefined;
    g.record.stats = stats;
  } else {
    g.record = { role: 'assistant', text: g.text, think: g.think || undefined, ts: Date.now(), stats };
    s?.messages.push(g.record);
  }
  if (s) s.updated = Date.now();
  persistSessions();

  const stick = nearBottom();
  if (g.think) {
    const d = thinkBlock(g.msg);
    d.querySelector('.think-body').textContent = g.think;
    if (g.text) d.open = false;
  }
  if (g.text) renderMarkdown(g.body, g.text, false);
  else g.body.innerHTML = `<span class="waiting">${g.think ? 'No answer after the reasoning. Try a higher max tokens.' : 'No reply.'}</span>`;
  renderAssistantFoot(g.msg, g.record);
  if (stick) scrollToBottom();
  showStats(stats);
  renderSessions();
  syncBusy();
}

function stop() {
  if (!gen) return;
  finalize('stopped');
  // The engine notices the closed socket on its next token send and stops.
  const old = socket;
  socket = null;
  if (old) { old.onclose = null; try { old.close(); } catch {} }
  setStatus('reconnecting');
  showEngine();
  syncBusy();
  connect(250);
}

// ── Composer ────────────────────────────────────────────────────────────────
function autosize() {
  el.input.style.height = 'auto';
  el.input.style.height = Math.min(el.input.scrollHeight, window.innerHeight * 0.4) + 'px';
}
el.input.addEventListener('input', autosize);
el.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    send();
  }
});
el.composer.addEventListener('submit', (e) => { e.preventDefault(); send(); });
el.stop.addEventListener('click', stop);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && gen) stop();
});

// ── Boot ────────────────────────────────────────────────────────────────────
applySettingsToUI();
renderPins();
renderAll();
showEngine();
connect();
el.input.focus();
