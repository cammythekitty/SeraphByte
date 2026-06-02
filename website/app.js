const socketUrl = "ws://127.0.0.1:8543";
let socket;

const chatLog = document.getElementById('chat-log');
const promptInput = document.getElementById('prompt-input');
const promptForm = document.getElementById('prompt-form');
const sendBtn = document.getElementById('send-btn');
const statusBadge = document.getElementById('status-badge');
const ctxLabel = document.getElementById('ctx-label');
const modelLine = document.getElementById('model-line');

marked.setOptions({ breaks: true, gfm: true });

let currentAssistantBubble = null; 
let currentRawText = '';           
let typingIndicator = null;        
let totalTokens = 0;

function nowTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function estimateTokens(text) {
    return Math.round(text.length / 4);
}

function updateCtx(text) {
    totalTokens += estimateTokens(text);
    if (ctxLabel) {
        ctxLabel.textContent = `ctx: ${totalTokens.toLocaleString()} / 8,192 tok`;
    }
}

function connectEngine() {
    socket = new WebSocket(socketUrl);

    socket.onopen = () => {
        statusBadge.textContent = "Connected";
        statusBadge.className = "status-connected";
        sendBtn.removeAttribute('disabled');
        if (modelLine) modelLine.textContent = "active_model · ws://127.0.0.1:8543";
    };

    socket.onmessage = (event) => {
        if (!currentAssistantBubble) {
            removeTypingIndicator();
            currentRawText = '';
            currentAssistantBubble = createAssistantBubble();
        }

        currentRawText += event.data;

        // Code block structural wrapper intercept guard
        let workingText = currentRawText;
        const backtickCount = (workingText.match(/```/g) || []).length;
        if (backtickCount % 2 !== 0) {
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

        statusBadge.textContent = "Disconnected";
        statusBadge.className = "status-disconnected";
        sendBtn.setAttribute('disabled', 'true');
        if (modelLine) modelLine.textContent = "offline · disconnected";

        setTimeout(connectEngine, 3000);
    };
}

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

// Emits timestamp + dynamic exact token validation counts per response chunk
function finalizeAssistantBubble(bubbleEl, rawText) {
    updateCtx(rawText);
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = `${nowTime()} · ~${estimateTokens(rawText)} tok`;
    bubbleEl.parentElement.appendChild(meta);
}

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

promptForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text || socket.readyState !== WebSocket.OPEN) return;

    createUserBubble(text);
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

connectEngine();