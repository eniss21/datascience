FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Train models during build (optional - remove if you want to mount pre-trained models)
RUN python -c "\
import sys; sys.path.insert(0, '.'); \
from config import *; \
from src.models.tfidf_classifier import TFIDFClassifier; \
from src.models.semantic_classifier import SemanticClassifier; \
from src.models.rnn_classifier import RNNIntentClassifier; \
tfidf = TFIDFClassifier(); tfidf.train(str(INTENTS_PATH)); tfidf.save(str(TFIDF_MODEL_PATH)); \
semantic = SemanticClassifier(); semantic.train(str(INTENTS_PATH)); semantic.save(str(SEMANTIC_MODEL_PATH)); \
rnn = RNNIntentClassifier(); rnn.train(str(INTENTS_PATH), epochs=100); rnn.save(str(RNN_MODEL_PATH)); \
print('All models trained!')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
