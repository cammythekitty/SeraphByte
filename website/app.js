// ─── Config ──────────────────────────────────────────────────────────────────
const SOCKET_URL = "ws://127.0.0.1:8543";
const CTX_MAX    = 8192;

// ─── DOM refs ────────────────────────────────────────────────────────────────
const chatLog         = document.getElementById('chat-log');
const promptInput     = document.getElementById('prompt-input');
const promptForm      = document.getElementById('prompt-form');
const sendBtn         = document.getElementById('send-btn');
const statusBadge     = document.getElementById('status-badge');
const ctxLabel        = document.getElementById('ctx-label');
const ctxBarFill      = document.getElementById('ctx-bar-fill');
const modelLine       = document.getElementById('model-line');
const modelSelect     = document.getElementById('model-select');
const memoryList      = document.getElementById('memory-list');
const statMessages    = document.getElementById('stat-messages');
const statTokens      = document.getElementById('stat-tokens');
const statCtxPct      = document.getElementById('stat-ctx-pct');
const statModel       = document.getElementById('stat-model');
const statConn        = document.getElementById('stat-conn');
const statLatency     = document.getElementById('stat-latency');

marked.setOptions({ breaks: true, gfm: true });

// ─── State ───────────────────────────────────────────────────────────────────
let socket;
let currentAssistantBubble = null;
let currentRawText         = '';
let typingIndicator        = null;
let totalTokens            = 0;
let messageCount           = 0;
let lastSendTime           = 0;

// ─── MemoryStore ─────────────────────────────────────────────────────────────
// Lightweight localStorage-backed store. Future: swap save/load for
// websocket RPCs to your Rust backend (e.g. ws MEMORY_SET / MEMORY_GET frames).
const MemoryStore = (() => {
    const KEY = 'seraph_memory_v1';

    function load() {
        try { return JSON.parse(localStorage.getItem(KEY)) || []; }
        catch { return []; }
    }

    function save(items) {
        localStorage.setItem(KEY, JSON.stringify(items));
    }

    return {
        getAll() { return load(); },
        add(text, tag = 'pin') {
            const items = load();
            const entry = { id: Date.now(), text: text.trim(), tag, ts: new Date().toISOString() };
            items.push(entry);
            save(items);
            return entry;
        },
        remove(id) {
            save(load().filter(e => e.id !== id));
        },
        clear() { save([]); },
        // Future hook: call this to broadcast memory to Rust backend
        // async sync(wsSocket) { wsSocket.send(JSON.stringify({ type:'MEMORY_SYNC', payload: load() })); }
    };
})();

// ─── Memory UI ───────────────────────────────────────────────────────────────
function renderMemoryList() {
    const items = MemoryStore.getAll();
    memoryList.innerHTML = '';
    if (items.length === 0) {
        memoryList.innerHTML = '<div class="mem-empty">No pinned memories yet.</div>';
        return;
    }
    items.forEach(entry => {
        const chip = document.createElement('div');
        chip.className = 'mem-chip';
        chip.innerHTML = `
            <span class="mem-icon">●</span>
            <span class="mem-text">${escapeHtml(entry.text)}</span>
            <button class="mem-remove" title="Remove" onclick="removeMemory(${entry.id})">✕</button>
        `;
        memoryList.appendChild(chip);
    });
}

function promptAddMemory(prefill = '') {
    const text = prompt('Pin to memory:', prefill);
    if (text && text.trim()) {
        MemoryStore.add(text.trim());
        renderMemoryList();
    }
}

function removeMemory(id) {
    MemoryStore.remove(id);
    renderMemoryList();
}

window.promptAddMemory = promptAddMemory;
window.removeMemory    = removeMemory;

// ─── Helpers ─────────────────────────────────────────────────────────────────
function nowTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function estimateTokens(text) {
    return Math.round(text.length / 4);
}

function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function updateCtx(text) {
    totalTokens += estimateTokens(text);
    const pct = Math.min(100, (totalTokens / CTX_MAX) * 100);
    if (ctxLabel)   ctxLabel.textContent = `ctx: ${totalTokens.toLocaleString()} / ${CTX_MAX.toLocaleString()} tok`;
    if (ctxBarFill) ctxBarFill.style.width = pct + '%';
    if (statTokens) statTokens.textContent = totalTokens.toLocaleString();
    if (statCtxPct) statCtxPct.textContent = Math.round(pct) + '%';
    // Warn visually when context is getting full
    if (ctxBarFill) {
        ctxBarFill.style.background = pct > 85 ? '#f87171' : pct > 60 ? '#fbbf24' : '';
    }
}

function updateMessageCount() {
    messageCount++;
    if (statMessages) statMessages.textContent = messageCount;
}

// ─── WebSocket ───────────────────────────────────────────────────────────────
function connectEngine() {
    socket = new WebSocket(SOCKET_URL);

    socket.onopen = () => {
        statusBadge.textContent = 'Connected';
        statusBadge.className   = 'status-connected';
        sendBtn.removeAttribute('disabled');
        modelLine.textContent   = `active_model · ${SOCKET_URL}`;
        if (statConn) { statConn.textContent = 'YES'; statConn.style.color = '#34d399'; }

        // Future: request model list from backend
        // socket.send(JSON.stringify({ type: 'LIST_MODELS' }));
    };

    socket.onmessage = (event) => {
        // Future: handle JSON frames (model list, memory sync ACK, etc.)
        // if (event.data[0] === '{') { handleControlFrame(JSON.parse(event.data)); return; }

        if (!currentAssistantBubble) {
            removeTypingIndicator();
            if (lastSendTime) {
                const latency = Date.now() - lastSendTime;
                if (statLatency) statLatency.textContent = `~${latency}ms`;
                lastSendTime = 0;
            }
            currentRawText = '';
            currentAssistantBubble = createAssistantBubble();
        }

        currentRawText += event.data;

        let workingText = currentRawText;
        if ((workingText.match(/```/g) || []).length % 2 !== 0) {
            workingText += '\n```';
        }
        currentAssistantBubble.innerHTML = marked.parse(workingText);
        chatLog.scrollTop = chatLog.scrollHeight;
    };

    socket.onclose = () => {
        if (currentAssistantBubble && currentRawText) {
            finalizeAssistantBubble(currentAssistantBubble, currentRawText);
            currentAssistantBubble = null;
            currentRawText = '';
        }

        statusBadge.textContent = 'Disconnected';
        statusBadge.className   = 'status-disconnected';
        sendBtn.setAttribute('disabled', 'true');
        modelLine.textContent   = `offline · disconnected`;
        if (statConn) { statConn.textContent = 'NO'; statConn.style.color = '#f87171'; }

        setTimeout(connectEngine, 3000);
    };
}

// ─── Bubble builders ─────────────────────────────────────────────────────────
function createAssistantBubble() {
    const wrapper = document.createElement('div');
    wrapper.className = 'msg-row-system msg-fade-in';

    const avatar = document.createElement('div');
    avatar.className = 'avatar-system';
    avatar.textContent = 'S';

    const col = document.createElement('div');
    col.className = 'msg-body';

    const body = document.createElement('div');
    body.className = 'bubble-system markdown-body';

    col.appendChild(body);
    wrapper.appendChild(avatar);
    wrapper.appendChild(col);
    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;
    updateMessageCount();

    return body;
}

function finalizeAssistantBubble(bubbleEl, rawText) {
    updateCtx(rawText);
    injectCodeCopyButtons(bubbleEl);

    const actions = document.createElement('div');
    actions.className = 'bubble-actions';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'bact';
    copyBtn.textContent = 'copy';
    copyBtn.onclick = () => {
        navigator.clipboard.writeText(rawText).then(() => {
            copyBtn.textContent = 'copied!';
            setTimeout(() => copyBtn.textContent = 'copy', 1500);
        });
    };

    const pinBtn = document.createElement('button');
    pinBtn.className = 'bact';
    pinBtn.textContent = 'pin';
    pinBtn.onclick = () => {
        const snippet = rawText.slice(0, 120).replace(/\n/g, ' ') + (rawText.length > 120 ? '…' : '');
        promptAddMemory(snippet);
    };

    actions.appendChild(copyBtn);
    actions.appendChild(pinBtn);

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = `${nowTime()} · ~${estimateTokens(rawText)} tok`;

    bubbleEl.parentElement.appendChild(actions);
    bubbleEl.parentElement.appendChild(meta);
}

function injectCodeCopyButtons(el) {
    el.querySelectorAll('pre').forEach(pre => {
        if (pre.querySelector('.code-copy-btn')) return;
        const btn = document.createElement('button');
        btn.className = 'code-copy-btn';
        btn.textContent = 'copy';
        btn.onclick = () => {
            const code = pre.querySelector('code');
            navigator.clipboard.writeText(code ? code.textContent : pre.textContent).then(() => {
                btn.textContent = 'copied!';
                setTimeout(() => btn.textContent = 'copy', 1500);
            });
        };
        pre.style.position = 'relative';
        pre.appendChild(btn);
    });
}

function createUserBubble(text) {
    const wrapper = document.createElement('div');
    wrapper.className = 'msg-row-user msg-fade-in';

    const avatar = document.createElement('div');
    avatar.className = 'avatar-user';
    avatar.textContent = 'U';

    const col = document.createElement('div');

    col.className = 'msg-body msg-body-right';
    const body = document.createElement('div');
    body.className = 'bubble-user';
    body.textContent = text;

    const actions = document.createElement('div');
    actions.className = 'bubble-actions bubble-actions-right';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'bact';
    copyBtn.textContent = 'copy';
    copyBtn.onclick = () => {
        navigator.clipboard.writeText(text).then(() => {
            copyBtn.textContent = 'copied!';
            setTimeout(() => copyBtn.textContent = 'copy', 1500);
        });
    };
    actions.appendChild(copyBtn);

    const meta = document.createElement('div');
    meta.className = 'msg-meta msg-meta-right';
    meta.textContent = nowTime();

    col.appendChild(body);
    col.appendChild(actions);
    col.appendChild(meta);
    wrapper.appendChild(col);
    wrapper.appendChild(avatar);
    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;

    updateCtx(text);
    updateMessageCount();
}

// ─── Typing indicator ────────────────────────────────────────────────────────
function showTypingIndicator() {
    typingIndicator = document.createElement('div');
    typingIndicator.className = 'msg-row-system msg-fade-in';

    const avatar = document.createElement('div');
    avatar.className = 'avatar-system';
    avatar.textContent = 'S';

    const bubble = document.createElement('div');
    bubble.className = 'bubble-system typing-bubble';
    bubble.innerHTML = '<span></span><span></span><span></span>';

    typingIndicator.appendChild(avatar);
    typingIndicator.appendChild(bubble);
    chatLog.appendChild(typingIndicator);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function removeTypingIndicator() {
    if (typingIndicator) { typingIndicator.remove(); typingIndicator = null; }
}

// ─── Sidebar ─────────────────────────────────────────────────────────────────
let sidebarCollapsed = false;

function switchPanel(btn, name) {
    ['memory','session','config'].forEach(p => {
        document.getElementById('panel-' + p).style.display = p === name ? '' : 'none';
    });
    document.querySelectorAll('.stab').forEach(t => {
        t.classList.toggle('active', t.dataset.panel === name);
    });
}

function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed;
    const sb = document.getElementById('sidebar');
    sb.classList.toggle('collapsed', sidebarCollapsed);
    const arrow = document.getElementById('collapse-arrow');
    const label = document.getElementById('collapse-label');
    arrow.textContent = sidebarCollapsed ? '▷' : '◁';
    if (label) label.style.display = sidebarCollapsed ? 'none' : '';
    document.querySelectorAll('.stab-label').forEach(el => el.style.display = sidebarCollapsed ? 'none' : '');
}

window.switchPanel    = switchPanel;
window.toggleSidebar  = toggleSidebar;

// ─── Form handlers ───────────────────────────────────────────────────────────
promptForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text || socket.readyState !== WebSocket.OPEN) return;

    createUserBubble(text);
    lastSendTime = Date.now();
    socket.send(text);

    currentAssistantBubble = null;
    currentRawText = '';
    showTypingIndicator();

    promptInput.value = '';
    promptInput.style.height = 'auto';
});

promptInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

promptInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        promptForm.requestSubmit();
    }
});

// ─── Init ────────────────────────────────────────────────────────────────────
renderMemoryList();
connectEngine();
