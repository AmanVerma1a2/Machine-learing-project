"""
Flask Sentiment Analysis Web Application - Multi-Model Comparison
==================================================================
A beautiful web app that shows predictions from 3 different ML algorithms
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import numpy as np

GoogleTranslator = None
TRANSLATOR_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    print("✓ deep_translator loaded successfully!")
except ImportError as e:
    print(f"⚠️  deep_translator not available: {e}")

# Common Hinglish/Hindi to English word mappings (fallback only)
HINGLISH_DICT = {
    'mei': 'I', 'acha': 'good', 'hu': 'am', 'main': 'I', 'bura': 'bad',
    'hoon': 'am', 'hun': 'am', 'hai': 'is', 'ho': 'are', 'hain': 'are',
    'bahut': 'very', 'nahi': 'not', 'bilkul': 'completely', 'kya': 'what',
    'kaise': 'how', 'kabhi': 'never', 'kab': 'when', 'yaar': 'friend',
    'bhai': 'brother', 'kharab': 'bad', 'dukhi': 'sad', 'khush': 'happy',
    'mast': 'great', 'bekar': 'useless', 'pyar': 'love', 'dil': 'heart',
    'mera': 'my', 'tera': 'your', 'uska': 'his', 'unka': 'their',
    'tum': 'you', 'aap': 'you', 'tu': 'you', 'wo': 'he/she',
}

app = Flask(__name__, template_folder='frontend', static_folder='frontend')

# Load vectorizer and encoder
print("Loading model files...")
with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Load all 4 models
models = {}
model_info = [
    ('models/model_multinomial_naive_bayes.pkl', 'Multinomial Naive Bayes', 'fa-brain'),
    ('models/model_logistic_regression.pkl', 'Logistic Regression', 'fa-chart-line'),
    ('models/model_linear_svm.pkl', 'Linear SVM', 'fa-microchip'),
    ('models/model_random_forest.pkl', 'Random Forest', 'fa-tree')
]

for filename, name, icon in model_info:
    try:
        with open(filename, 'rb') as f:
            models[name] = {'model': pickle.load(f), 'icon': icon}
        print(f"✓ {name} loaded successfully!")
    except FileNotFoundError:
        print(f"⚠️  {filename} not found - run train_model_comparison.py first")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

print(f"✓ {len(models)} models loaded successfully!")


def fallback_word_replace(text):
    """Simple fallback: replace common Hinglish/Hindi words with English."""
    words = re.findall(r"\b\w+\b", text.lower())
    has_hinglish = any(w in HINGLISH_DICT for w in words)
    
    if not has_hinglish:
        return text
    
    result = text.lower()
    for hindi_word, eng_word in HINGLISH_DICT.items():
        result = re.sub(r'\b' + hindi_word + r'\b', eng_word, result)
    
    return result


def translate_to_english(text):
    """Translate any input to English before preprocessing."""
    if not text or not text.strip():
        return text

    original_text = text.strip()
    
    # Strategy 1: Try Google Translator with auto-detect
    if TRANSLATOR_AVAILABLE:
        try:
            print(f"[Translation] Auto-detecting and translating: {original_text[:50]}")
            translator = GoogleTranslator(source='auto', target='en')
            result = translator.translate(original_text)
            if result and result.strip().lower() != original_text.lower():
                print(f"[Translation] AUTO success: {result[:50]}")
                return result.strip()
        except Exception as e:
            print(f"[Translation] Auto-detect failed: {e}")
        
        # Strategy 2: Try explicitly as Hindi
        try:
            print(f"[Translation] Trying Hindi source...")
            translator = GoogleTranslator(source='hi', target='en')
            result = translator.translate(original_text)
            if result and result.strip().lower() != original_text.lower():
                print(f"[Translation] HINDI source success: {result[:50]}")
                return result.strip()
        except Exception as e:
            print(f"[Translation] Hindi source failed: {e}")
    
    # Strategy 3: Fallback to word replacement for Hinglish
    print(f"[Translation] Using Hinglish fallback mapping")
    fallback_result = fallback_word_replace(original_text)
    if fallback_result.lower() != original_text.lower():
        print(f"[Translation] Fallback success: {fallback_result[:50]}")
        return fallback_result
    
    print(f"[Translation] No translation applied")
    return original_text

def clean_text(text):
    """Clean input text - matches training preprocessing"""
    if not text:
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment from user input using all 3 models"""
    try:
        # Get input text
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({
                'error': 'Please enter some text!'
            }), 400

        translated_text = translate_to_english(text)
        
        # Clean text
        cleaned_text = clean_text(translated_text)
        
        if not cleaned_text:
            return jsonify({
                'error': 'No valid text found after cleaning!'
            }), 400
        
        # Vectorize
        text_tfidf = tfidf.transform([cleaned_text])
        
        # Get predictions from all models
        predictions = []
        
        for name, model_data in models.items():
            try:
                model = model_data['model']
                icon = model_data['icon']
                
                # Predict
                pred = model.predict(text_tfidf)[0]
                sentiment = encoder.classes_[pred]
                
                # Calculate confidence
                try:
                    if hasattr(model, 'predict_proba'):
                        # For models with probability support
                        proba = model.predict_proba(text_tfidf)[0]
                        confidence = max(proba) * 100
                    elif hasattr(model, 'decision_function'):
                        # For SVM
                        decision_scores = model.decision_function(text_tfidf)[0]
                        confidence = float(max(abs(decision_scores))) * 10
                        confidence = min(confidence, 99.9)
                    else:
                        confidence = 85.0  # Default
                except:
                    confidence = 85.0
                
                predictions.append({
                    'model': name,
                    'sentiment': sentiment.capitalize(),
                    'confidence': f"{confidence:.1f}",
                    'icon': icon
                })
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        return jsonify({
            'predictions': predictions,
            'original_text': text,
            'translated_text': translated_text,
            'cleaned_text': cleaned_text
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error processing request: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 Starting Flask Sentiment Analysis App")
    print("=" * 70)
    print("\n📱 Open in browser: http://127.0.0.1:5000")
    print("Press CTRL+C to quit\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
