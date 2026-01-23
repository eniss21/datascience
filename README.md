# 🤖 NLP E-commerce Chatbot

An intelligent customer support chatbot for e-commerce platforms powered by multiple NLP models. The chatbot can understand customer intents and provide relevant responses using three different classification approaches.

## ✨ Features

- **Multiple ML Models**: Choose between TF-IDF, Semantic (Sentence Transformers), or RNN (BiLSTM with Attention)
- **28 Intent Categories**: Covers common e-commerce customer queries
- **REST API**: FastAPI-based backend with OpenAPI documentation
- **Web Interface**: Simple chat UI for testing
- **Data Augmentation**: Improved training with automated text augmentation
- **Docker Support**: Easy deployment with containerization

## 🏗️ Project Structure

```
nlp-chatbot/
├── config.py                 # Configuration settings
├── Dockerfile               # Docker containerization
├── requirements.txt         # Python dependencies
├── data/
│   └── intents.json        # Training data with patterns and responses
├── models/                  # Trained model files (generated)
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI application
│   ├── chatbot/
│   │   ├── intent_handler.py
│   │   └── response_generator.py
│   ├── models/
│   │   ├── tfidf_classifier.py    # TF-IDF + Logistic Regression
│   │   ├── semantic_classifier.py  # Sentence Transformers
│   │   └── rnn_classifier.py       # BiLSTM with Attention
│   └── preprocessing/
│       ├── text_processor.py       # Text normalization
│       └── data_augmentation.py    # Training data augmentation
└── static/
    ├── index.html          # Chat interface
    ├── script.js           # Frontend logic
    └── styles.css          # Styling
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12** (required - some packages like gensim don't support Python 3.13+)
- pip or conda

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd nlp-chatbot
   ```

2. **Create a virtual environment with Python 3.12**

   ```bash
   # macOS (using Homebrew Python 3.12)
   /opt/homebrew/bin/python3.12 -m venv .venv
   source .venv/bin/activate

   # Linux/Windows (if python3.12 is in PATH)
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (required for text processing)

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
   ```

5. **Train the models**

   ```bash
   python -c "
   import sys
   sys.path.insert(0, '.')
   from config import *

   # Train TF-IDF
   from src.models.tfidf_classifier import TFIDFClassifier
   tfidf = TFIDFClassifier()
   tfidf.train(str(INTENTS_PATH))
   tfidf.save(str(TFIDF_MODEL_PATH))

   # Train Semantic
   from src.models.semantic_classifier import SemanticClassifier
   semantic = SemanticClassifier()
   semantic.train(str(INTENTS_PATH))
   semantic.save(str(SEMANTIC_MODEL_PATH))

   # Train RNN
   from src.models.rnn_classifier import RNNIntentClassifier
   rnn = RNNIntentClassifier()
   rnn.train(str(INTENTS_PATH), epochs=EPOCHS)
   rnn.save(str(RNN_MODEL_PATH))

   print('All models trained!')
   "
   ```

6. **Start the server**

   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. **Open the chat interface**

   Navigate to http://localhost:8000 in your browser

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t nlp-chatbot .

# Run the container
docker run -p 8000:8000 nlp-chatbot
```

## 📡 API Endpoints

### Chat Endpoint

```http
POST /chat
Content-Type: application/json

{
    "message": "Where is my order?",
    "model_type": "semantic"  // Optional: "tfidf", "semantic", or "rnn"
}
```

**Response:**

```json
{
  "intent": "order_status",
  "confidence": 0.95,
  "response": "I'd be happy to help you track your order! Please provide your order number.",
  "is_fallback": false,
  "model_type": "semantic"
}
```

### Debug Endpoint

```http
POST /chat/debug
```

Returns detailed scoring information for all intents.

### Health Check

```http
GET /health
```

Returns status of loaded models.

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧠 Model Comparison

| Model        | Accuracy | Speed     | Best For                           |
| ------------ | -------- | --------- | ---------------------------------- |
| **TF-IDF**   | ~74%     | ⚡ Fast   | Quick responses, keyword matching  |
| **Semantic** | ~99%     | 🐢 Slower | Best accuracy, understands meaning |
| **RNN**      | ~87%     | 🚀 Medium | Balance of speed and accuracy      |

### Model Details

- **TF-IDF Classifier**: Uses word and character-level TF-IDF features with Logistic Regression. Good for typo tolerance.
- **Semantic Classifier**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) to understand semantic meaning. "talk to agent" matches "speak with representative".
- **RNN Classifier**: BiLSTM with attention mechanism. Trained with data augmentation for better generalization.

## 📋 Available Intents

The chatbot recognizes **28 different intents** for e-commerce customer support:

| Intent                 | Description       | Example Queries                                    |
| ---------------------- | ----------------- | -------------------------------------------------- |
| `greeting`             | Welcome messages  | "Hello", "Hi there", "Good morning"                |
| `goodbye`              | Farewell messages | "Bye", "See you later", "Thanks, goodbye"          |
| `thanks`               | Appreciation      | "Thank you", "Thanks for your help"                |
| `order_status`         | Track orders      | "Where is my order?", "Track my package"           |
| `order_cancel`         | Cancel orders     | "Cancel my order", "I want to cancel"              |
| `order_modify`         | Change orders     | "Change my order", "Update shipping address"       |
| `shipping_info`        | Delivery details  | "Shipping options", "How long does delivery take?" |
| `return_request`       | Return items      | "I want to return this", "Return policy"           |
| `refund_status`        | Check refunds     | "Where is my refund?", "Refund status"             |
| `product_availability` | Stock check       | "Is this in stock?", "When will it be available?"  |
| `product_info`         | Product details   | "Tell me about this product", "Product specs"      |
| `payment_methods`      | Payment options   | "What payment methods do you accept?"              |
| `payment_issue`        | Payment problems  | "Payment failed", "Card declined"                  |
| `discount_code`        | Promo codes       | "Do you have any coupons?", "Discount code"        |
| `account_help`         | Account issues    | "Reset password", "Can't login"                    |
| `contact_human`        | Live agent        | "Talk to a human", "Speak to agent"                |
| `store_hours`          | Business hours    | "What are your hours?", "When do you open?"        |
| `warranty`             | Warranty info     | "Warranty policy", "Is this covered?"              |
| `gift_card`            | Gift cards        | "Buy gift card", "Check gift card balance"         |
| `size_guide`           | Sizing help       | "Size chart", "What size should I get?"            |
| `rewards_program`      | Loyalty program   | "Loyalty points", "Rewards program"                |
| `complaint`            | File complaints   | "I have a complaint", "Bad experience"             |
| `positive_feedback`    | Compliments       | "Great service!", "Love your store"                |
| `bulk_orders`          | Wholesale         | "Bulk order discount", "Wholesale pricing"         |
| `subscription`         | Subscriptions     | "Cancel subscription", "Manage subscription"       |
| `technical_support`    | Tech issues       | "Website not working", "App crash"                 |
| `privacy_security`     | Data privacy      | "Delete my data", "Privacy policy"                 |
| `newsletter`           | Email signup      | "Subscribe to newsletter", "Unsubscribe"           |
| `fallback`             | Unknown queries   | Triggered when confidence is low                   |

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Confidence thresholds per model
CONFIDENCE_THRESHOLDS = {
    "tfidf": 0.35,
    "semantic": 0.55,
    "rnn": 0.40
}

# RNN training parameters
EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 20

# Semantic model
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
```

## 🔧 Adding New Intents

1. Edit `data/intents.json`:

   ```json
   {
     "tag": "new_intent",
     "patterns": ["Example pattern 1", "Example pattern 2", "..."],
     "responses": ["Response 1", "Response 2"]
   }
   ```

2. Retrain the models (see Installation step 5)

## 🧪 Testing

```bash
# Run tests
pytest

# Test a specific model
python -c "
from src.models.semantic_classifier import SemanticClassifier
from config import SEMANTIC_MODEL_PATH

model = SemanticClassifier()
model.load(str(SEMANTIC_MODEL_PATH))

queries = ['hello', 'where is my order', 'I want a refund']
for q in queries:
    intent, conf = model.predict(q)
    print(f'{q} -> {intent} ({conf:.2f})')
"
```

## 📊 Performance Tips

- **For production**: Use the Semantic model for best accuracy
- **For high traffic**: Use TF-IDF model for faster response times
- **GPU acceleration**: RNN model benefits from CUDA if available

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
