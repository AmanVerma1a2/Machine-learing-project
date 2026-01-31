// DOM Elements
const textInput = document.getElementById('textInput');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const multiModelResults = document.getElementById('multiModelResults');
const originalText = document.getElementById('originalText');
const processedText = document.getElementById('processedText');
const errorText = document.getElementById('errorText');
const exampleButtons = document.querySelectorAll('.btn-example');

// Character counter
textInput.addEventListener('input', () => {
    const count = textInput.value.length;
    charCount.textContent = count;
});

// Example buttons
exampleButtons.forEach(button => {
    button.addEventListener('click', () => {
        const exampleText = button.getAttribute('data-text');
        textInput.value = exampleText;
        charCount.textContent = exampleText.length;
        textInput.focus();
    });
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    
    // Validate input
    if (!text) {
        showError('Please enter some text to analyze!');
        return;
    }
    
    // Hide previous results/errors
    resultSection.classList.remove('active');
    errorSection.classList.remove('active');
    
    // Show loading
    loading.classList.add('active');
    analyzeBtn.disabled = true;
    
    try {
        // Send request to Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        // Hide loading
        loading.classList.remove('active');
        analyzeBtn.disabled = false;
        
        if (response.ok) {
            // Show result
            displayResult(data);
        } else {
            // Show error
            showError(data.error || 'An error occurred while analyzing the text.');
        }
        
    } catch (error) {
        // Hide loading
        loading.classList.remove('active');
        analyzeBtn.disabled = false;
        
        // Show error
        showError('Failed to connect to the server. Please make sure the server is running.');
        console.error('Error:', error);
    }
});

// Display result - Updated for multiple models
function displayResult(data) {
    // Clear previous results
    multiModelResults.innerHTML = '';
    
    // Check if we have predictions array
    if (data.predictions && data.predictions.length > 0) {
        // Create card for each model
        data.predictions.forEach((pred, index) => {
            const modelCard = createModelCard(pred, index);
            multiModelResults.appendChild(modelCard);
        });
    } else {
        // Fallback for single model response (backwards compatibility)
        const singlePred = {
            model: 'Model',
            sentiment: data.sentiment,
            confidence: data.confidence,
            icon: 'fa-robot'
        };
        const modelCard = createModelCard(singlePred, 0);
        multiModelResults.appendChild(modelCard);
    }
    
    // Set text details
    originalText.textContent = data.original_text;
    processedText.textContent = data.cleaned_text || 'N/A';
    
    // Show result section
    resultSection.classList.add('active');
    
    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Create model card
function createModelCard(pred, index) {
    const sentiment = pred.sentiment.toLowerCase();
    const confidence = parseFloat(pred.confidence);
    
    // Determine sentiment icon
    let sentimentIcon = 'fa-meh';
    switch(sentiment) {
        case 'positive':
            sentimentIcon = 'fa-smile';
            break;
        case 'negative':
            sentimentIcon = 'fa-frown';
            break;
        case 'neutral':
            sentimentIcon = 'fa-meh';
            break;
    }
    
    // Create card element
    const card = document.createElement('div');
    card.className = `model-card ${sentiment}`;
    card.style.animationDelay = `${index * 0.1}s`;
    
    card.innerHTML = `
        <div class="model-header">
            <i class="fas ${pred.icon}"></i>
            <h4>${pred.model}</h4>
        </div>
        <div class="model-body">
            <div class="sentiment-badge ${sentiment}">
                <i class="fas ${sentimentIcon}"></i>
                <span>${pred.sentiment}</span>
            </div>
            <div class="confidence-meter">
                <label>Confidence</label>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${confidence}%"></div>
                </div>
                <span class="confidence-text">${confidence}%</span>
            </div>
        </div>
    `;
    
    return card;
}

// Show error
function showError(message) {
    errorText.textContent = message;
    errorSection.classList.add('active');
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        errorSection.classList.remove('active');
    }, 5000);
}

// Allow Enter key to submit (with Shift+Enter for new line)
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        analyzeBtn.click();
    }
});

// Add smooth scroll behavior
document.documentElement.style.scrollBehavior = 'smooth';
