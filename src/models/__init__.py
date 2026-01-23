from .tfidf_classifier import TFIDFClassifier
from .semantic_classifier import SemanticClassifier

# Optional imports - only import if torch is available
try:
    from .rnn_classifier import RNNIntentClassifier, IntentClassifier
except ImportError:
    # torch not installed, RNN classifier not available
    RNNIntentClassifier = None
    IntentClassifier = None
