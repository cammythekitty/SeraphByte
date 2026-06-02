const socketUrl = "ws://127.0.0.1:8543";
let socket;

const chatLog = document.getElementById('chat-log');
const promptInput = document.getElementById('prompt-input');
const promptForm = document.getElementById('prompt-form');
const sendBtn = document.getElementById('send-btn');
const statusBadge = document.getElementById('status-badge');

let currentAssistantBubble = null;

// Initialize connection to the local Rust engine
function connectEngine() {
    socket = new WebSocket(socketUrl);

    socket.onopen = () => {
        statusBadge.textContent = "Connected";
        statusBadge.className = "status-connected";
        sendBtn.removeAttribute('disabled');
    };

    socket.onmessage = (event) => {
        // If there isn't an active assistant block streaming, spin one up
        if (!currentAssistantBubble) {
            currentAssistantBubble = createChatBubble('assistant');
        }
        
        // Append tokens dynamically to the active response container
        currentAssistantBubble.innerText += event.data;
        chatLog.scrollTop = chatLog.scrollHeight;
    };

    socket.onclose = () => {
        statusBadge.textContent = "Disconnected";
        statusBadge.className = "status-disconnected";
        sendBtn.setAttribute('disabled', 'true');
        
        // Attempt an automatic reconnection sequence every 3 seconds
        setTimeout(connectEngine, 3000);
    };
}

// Factory function to render messaging bubbles inside the DOM
function createChatBubble(sender) {
    const wrapper = document.createElement('div');
    wrapper.className = `flex gap-3 ${sender === 'user' ? 'justify-end' : 'max-w-2xl'}`;

    const avatar = document.createElement('div');
    avatar.className = sender === 'user' ? 'avatar-user' : 'avatar-system';
    avatar.textContent = sender === 'user' ? 'U' : 'S';

    const body = document.createElement('div');
    body.className = sender === 'user' ? 'bubble-user whitespace-pre-wrap' : 'bubble-system whitespace-pre-wrap';

    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;

    return body; // Returns reference so the token loop can append straight to it
}

// Prompt Submission handler
promptForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text || socket.readyState !== WebSocket.OPEN) return;

    // Render the user's message bubble
    createChatBubble('user').textContent = text;
    
    // Send the payload directly to the Rust WebSocket Gateway
    socket.send(text);
    
    // Reset flags for the upcoming text generation pipeline
    currentAssistantBubble = null;
    promptInput.value = '';
    promptInput.style.height = 'auto';
});

// Auto-expand input box as text wraps around long entries
promptInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Intercept standalone Enter triggers, allowing Shift+Enter for clean lines
promptInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        promptForm.requestSubmit();
    }
});

// Fire up network event triggers
connectEngine();