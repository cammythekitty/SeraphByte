const socketUrl = "ws://127.0.0.1:8543";
let socket;

const chatLog = document.getElementById('chat-log');
const promptInput = document.getElementById('prompt-input');
const promptForm = document.getElementById('prompt-form');
const sendBtn = document.getElementById('send-btn');
const statusBadge = document.getElementById('status-badge');
const ctxLabel = document.getElementById('ctx-label');

// Configure marked engine for clean, standard Github-Flavored Markdown parsing
marked.setOptions({ breaks: true, gfm: true });

let currentAssistantBubble = null; // Target .markdown-body DOM reference
let currentRawText = '';           // Running text stream accumulator
let typingIndicator = null;        // Active typing indicator row
let totalTokens = 0;

function nowTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function estimateTokens(text) {
    // Standard local metric approximation: ~4 characters per hardware token
    return Math.round(text.length / 4);
}

function updateCtx(text) {
    totalTokens += estimateTokens(text);
    if (ctxLabel) {
        ctxLabel.textContent = `ctx: ${totalTokens.toLocaleString()} / 8192 tok`;
    }
}

// Initialize connection network layout to the local Rust server
function connectEngine() {
    socket = new WebSocket(socketUrl);

    socket.onopen = () => {
        statusBadge.textContent = "Connected";
        statusBadge.className = "status-connected";
        sendBtn.removeAttribute('disabled');
    };

    socket.onmessage = (event) => {
        // First token event: remove loading dots and spawn real bubble container
        if (!currentAssistantBubble) {
            removeTypingIndicator();
            currentRawText = '';
            currentAssistantBubble = createAssistantBubble();
        }

        // Accumulate raw string tokens from the websocket channel
        currentRawText += event.data;

        // Dynamic Stream Guard: If backtick strings are odd, append a temporary closing block 
        // to keep the browser DOM from shattering while the model is typing code arrays
        let workingText = currentRawText;
        const backtickCount = (workingText.match(/```/g) || []).length;
        if (backtickCount % 2 !== 0) {
            workingText += '\n```'; 
        }

        // Safely parse and render into clean HTML layouts
        currentAssistantBubble.innerHTML = marked.parse(workingText);
        chatLog.scrollTop = chatLog.scrollHeight;
    };

    socket.onclose = () => {
        // Network teardown: finalize remaining stream fragments safely
        if (currentAssistantBubble && currentRawText) {
            finalizeAssistantBubble(currentAssistantBubble, currentRawText);
            currentAssistantBubble = null;
            currentRawText = '';
        }

        statusBadge.textContent = "Disconnected";
        statusBadge.className = "status-disconnected";
        sendBtn.setAttribute('disabled', 'true');

        // Retry connection sequence loops every 3 seconds if backend drops
        setTimeout(connectEngine, 3000);
    };
}

// Build system message rows featuring the new layout designs
function createAssistantBubble() {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex gap-3 max-w-2xl msg-row-system msg-fade-in';

    const avatar = document.createElement('div');
    avatar.className = 'avatar-system';
    avatar.textContent = 'S';

    const body = document.createElement('div');
    body.className = 'bubble-system markdown-body';

    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;

    return body;
}

// Finalizes chat text arrays with timestamps and metrics metrics
function finalizeAssistantBubble(bubbleEl, rawText) {
    updateCtx(rawText);
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = `${nowTime()} · ~${estimateTokens(rawText)} tok`;
    bubbleEl.parentElement.appendChild(meta);
}

// Renders the moving structural waiting animation bubble
function showTypingIndicator() {
    typingIndicator = document.createElement('div');
    typingIndicator.className = 'flex gap-3 max-w-2xl msg-row-system msg-fade-in';

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
    if (typingIndicator) {
        typingIndicator.remove();
        typingIndicator = null;
    }
}

// User text bubble construction factory
function createUserBubble(text) {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex gap-3 justify-end msg-fade-in';

    const avatar = document.createElement('div');
    avatar.className = 'avatar-user';
    avatar.textContent = 'U';

    const col = document.createElement('div');

    const body = document.createElement('div');
    body.className = 'bubble-user';
    body.textContent = text;

    const meta = document.createElement('div');
    meta.className = 'msg-meta msg-meta-right';
    meta.textContent = nowTime();

    col.appendChild(body);
    col.appendChild(meta);
    wrapper.appendChild(col);
    wrapper.appendChild(avatar);
    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;

    updateCtx(text);
}

// Intercept form submissions
promptForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text || socket.readyState !== WebSocket.OPEN) return;

    createUserBubble(text);
    socket.send(text);

    // Clean streaming registers and flip typing indicator live
    currentAssistantBubble = null;
    currentRawText = '';
    showTypingIndicator();

    promptInput.value = '';
    promptInput.style.height = 'auto';
});

// Dynamic field adjustment as text lengths scale
promptInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

// Handle line splits cleanly
promptInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        promptForm.requestSubmit();
    }
});

// Fire up event networking pipeline
connectEngine();