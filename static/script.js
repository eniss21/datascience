// ========================================
// E-commerce Support Chatbot - JavaScript
// ========================================

// Configuration
const API_BASE_URL = 'http://localhost:8000';
let currentModel = 'tfidf';
let showDebug = false;
let hasMessages = false;

// DOM Elements
const elements = {
    chatMessages: null,
    userInput: null,
    sendButton: null,
    sidebar: null,
    overlay: null,
    debugDrawer: null,
    debugToggle: null,
    debugContent: null,
    statusIndicator: null,
    welcomeSection: null,
    mobileModelBadge: null
};

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    loadModelPreference();
    setupEventListeners();
    checkAPIHealth();
    elements.userInput.focus();
});

function initializeElements() {
    elements.chatMessages = document.getElementById('chatMessages');
    elements.userInput = document.getElementById('userInput');
    elements.sendButton = document.getElementById('sendButton');
    elements.sidebar = document.getElementById('sidebar');
    elements.overlay = document.getElementById('overlay');
    elements.debugDrawer = document.getElementById('debugDrawer');
    elements.debugToggle = document.getElementById('debugToggle');
    elements.debugContent = document.getElementById('debugContent');
    elements.statusIndicator = document.getElementById('statusIndicator');
    elements.welcomeSection = document.getElementById('welcomeSection');
    elements.mobileModelBadge = document.getElementById('mobileModelBadge');
}

function setupEventListeners() {
    // Input enter key
    elements.userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Model selection
    document.querySelectorAll('input[name="model"]').forEach(radio => {
        radio.addEventListener('change', function() {
            switchModel(this.value);
        });
    });

    // Debug toggle
    elements.debugToggle.addEventListener('change', function() {
        showDebug = this.checked;
        if (showDebug) {
            elements.debugDrawer.classList.add('open');
        } else {
            elements.debugDrawer.classList.remove('open');
        }
    });

    // Close sidebar on escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeSidebar();
            closeDebug();
        }
    });
}

// ========================================
// Model Management
// ========================================

function loadModelPreference() {
    const savedModel = localStorage.getItem('chatbotModel') || 'tfidf';
    currentModel = savedModel;

    // Update radio button
    const radio = document.querySelector(`input[name="model"][value="${savedModel}"]`);
    if (radio) {
        radio.checked = true;
    }

    updateModelBadge();
}

function switchModel(model) {
    currentModel = model;
    localStorage.setItem('chatbotModel', model);
    updateModelBadge();

    // Show notification
    addSystemMessage(`Switched to ${getModelDisplayName(model)} model`);
}

function getModelDisplayName(model) {
    const names = {
        'tfidf': 'TF-IDF',
        'semantic': 'Semantic',
        'rnn': 'RNN'
    };
    return names[model] || model;
}

function updateModelBadge() {
    if (elements.mobileModelBadge) {
        elements.mobileModelBadge.textContent = getModelDisplayName(currentModel);
    }
}

// ========================================
// API Health Check
// ========================================

async function checkAPIHealth() {
    const statusText = elements.statusIndicator.querySelector('.status-text');

    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        elements.statusIndicator.classList.remove('offline');
        elements.statusIndicator.classList.add('online');

        // Check if any models are loaded (new format: models_loaded dict, old format: model_loaded bool)
        const hasModels = data.models_loaded
            ? Object.keys(data.models_loaded).length > 0
            : data.model_loaded;

        if (hasModels) {
            const modelCount = data.models_loaded ? Object.keys(data.models_loaded).length : 1;
            statusText.textContent = `Connected (${modelCount} models)`;
        } else {
            statusText.textContent = 'No models loaded';
            addSystemMessage('Warning: No models loaded on server. Make sure the API is running and models are trained.');
        }
    } catch (error) {
        elements.statusIndicator.classList.remove('online');
        elements.statusIndicator.classList.add('offline');
        statusText.textContent = 'Disconnected';
        addSystemMessage('Cannot connect to API server. Make sure it\'s running on ' + API_BASE_URL);
        console.error('API health check failed:', error);
    }
}

// ========================================
// Message Handling
// ========================================

function sendQuickMessage(message) {
    elements.userInput.value = message;
    sendMessage();
}

async function sendMessage() {
    const message = elements.userInput.value.trim();

    if (!message) return;

    // Hide welcome section on first message
    if (!hasMessages && elements.welcomeSection) {
        elements.welcomeSection.style.display = 'none';
        hasMessages = true;
    }

    // Disable input
    setInputState(false);

    // Add user message
    addUserMessage(message);

    // Clear input
    elements.userInput.value = '';

    // Show typing indicator
    const typingId = addTypingIndicator();

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message, model_type: currentModel })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Remove typing indicator
        removeTypingIndicator(typingId);

        // Add bot response
        addBotMessage(data.response, data.intent, data.confidence, data.is_fallback);

        // Fetch debug info if enabled
        if (showDebug) {
            await fetchDebugInfo(message);
        }

    } catch (error) {
        removeTypingIndicator(typingId);
        addBotMessage(
            'Sorry, I encountered an error. Please make sure the API server is running.',
            'error',
            0,
            true
        );
        console.error('Error sending message:', error);
    } finally {
        setInputState(true);
        elements.userInput.focus();
    }
}

function setInputState(enabled) {
    elements.sendButton.disabled = !enabled;
    elements.userInput.disabled = !enabled;
}

// ========================================
// Message Display
// ========================================

function addUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';

    const timestamp = formatTime(new Date());

    messageDiv.innerHTML = `
        <div class="message-bubble">
            <p class="message-text">${escapeHtml(message)}</p>
            <div class="message-meta">
                <span class="timestamp">${timestamp}</span>
            </div>
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addBotMessage(message, intent, confidence, isFallback) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';

    const timestamp = formatTime(new Date());
    const confidenceClass = getConfidenceClass(confidence);
    const confidencePercent = (confidence * 100).toFixed(1);

    let metaContent = `
        <span class="intent-badge">${intent}</span>
        <span class="confidence-badge ${confidenceClass}">${confidencePercent}%</span>
        <span class="timestamp">${timestamp}</span>
    `;

    let fallbackAlert = '';
    if (isFallback) {
        fallbackAlert = `
            <div class="fallback-alert">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                Low confidence - fallback response
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-bubble">
            <p class="message-text">${escapeHtml(message)}</p>
            <div class="message-meta">
                ${metaContent}
            </div>
            ${fallbackAlert}
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addSystemMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';

    messageDiv.innerHTML = `
        <div class="message-bubble">
            <p class="message-text">${escapeHtml(message)}</p>
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.id = 'typing-indicator';

    messageDiv.innerHTML = `
        <div class="message-bubble">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return 'typing-indicator';
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

// ========================================
// Debug Panel
// ========================================

async function fetchDebugInfo(message) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat/debug`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message, model_type: currentModel })
        });

        if (response.ok) {
            const data = await response.json();
            displayDebugInfo(data);
        }
    } catch (error) {
        console.error('Error fetching debug info:', error);
    }
}

function displayDebugInfo(data) {
    elements.debugContent.innerHTML = `
        <pre>${JSON.stringify({
            message: data.message,
            selected_intent: data.selected_intent,
            confidence: data.confidence.toFixed(4),
            top_intents: data.top_intents,
            response: data.response
        }, null, 2)}</pre>
    `;
}

function closeDebug() {
    elements.debugDrawer.classList.remove('open');
    elements.debugToggle.checked = false;
    showDebug = false;
}

// ========================================
// Sidebar
// ========================================

function toggleSidebar() {
    elements.sidebar.classList.toggle('open');
    elements.overlay.classList.toggle('visible');
}

function closeSidebar() {
    elements.sidebar.classList.remove('open');
    elements.overlay.classList.remove('visible');
}

// ========================================
// Chat Actions
// ========================================

function clearChat() {
    if (!confirm('Clear all messages?')) return;

    hasMessages = false;

    elements.chatMessages.innerHTML = `
        <div class="welcome-section" id="welcomeSection">
            <div class="welcome-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
                    <line x1="9" y1="9" x2="9.01" y2="9"/>
                    <line x1="15" y1="9" x2="15.01" y2="9"/>
                </svg>
            </div>
            <h2 class="welcome-title">How can I help you today?</h2>
            <p class="welcome-subtitle">I'm your e-commerce support assistant. Ask me about orders, shipping, returns, and more.</p>

            <div class="quick-actions">
                <button class="quick-action" onclick="sendQuickMessage('Where is my order?')">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="1" y="3" width="15" height="13"/>
                        <polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/>
                        <circle cx="5.5" cy="18.5" r="2.5"/>
                        <circle cx="18.5" cy="18.5" r="2.5"/>
                    </svg>
                    Track Order
                </button>
                <button class="quick-action" onclick="sendQuickMessage('How do I return an item?')">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="1 4 1 10 7 10"/>
                        <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
                    </svg>
                    Return Item
                </button>
                <button class="quick-action" onclick="sendQuickMessage('I need help with payment')">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="1" y="4" width="22" height="16" rx="2" ry="2"/>
                        <line x1="1" y1="10" x2="23" y2="10"/>
                    </svg>
                    Payment Help
                </button>
                <button class="quick-action" onclick="sendQuickMessage('I have a complaint')">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                    File Complaint
                </button>
            </div>
        </div>
    `;

    elements.welcomeSection = document.getElementById('welcomeSection');
}

// ========================================
// Utility Functions
// ========================================

function getConfidenceClass(confidence) {
    if (confidence >= 0.7) return 'confidence-high';
    if (confidence >= 0.4) return 'confidence-medium';
    return 'confidence-low';
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
