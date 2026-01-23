import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Model settings
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.joblib"
RNN_MODEL_PATH = MODELS_DIR / "rnn_model.pt"
SEMANTIC_MODEL_PATH = MODELS_DIR / "semantic_model.pkl"
INTENTS_PATH = DATA_DIR / "intents.json"

# Preprocessing settings
MAX_SEQUENCE_LENGTH = 30
MIN_CONFIDENCE_THRESHOLD = 0.3

# Model-specific confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "tfidf": 0.35,
    "semantic": 0.55,
    "rnn": 0.40
}

# TF-IDF settings
TFIDF_MAX_FEATURES = 2000

# Semantic model settings
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_USE_PREPROCESSING = False

# RNN model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.4
LEARNING_RATE = 0.001
EPOCHS = 150
BATCH_SIZE = 16
RNN_USE_AUGMENTATION = True
EARLY_STOP_PATIENCE = 20

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
