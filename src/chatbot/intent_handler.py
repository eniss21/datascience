from pathlib import Path
from typing import Tuple, Optional, Literal
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.tfidf_classifier import TFIDFClassifier
from src.models.semantic_classifier import SemanticClassifier

# Optional import for RNN - only if torch is available
try:
    from src.models.rnn_classifier import RNNIntentClassifier
except ImportError:
    RNNIntentClassifier = None


class IntentHandler:
    """Handles intent classification for the chatbot."""

    def __init__(
        self,
        model_type: Literal["tfidf", "rnn", "semantic"] = "tfidf",
        confidence_threshold: float = 0.3
    ):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model: Optional[TFIDFClassifier | RNNIntentClassifier | SemanticClassifier] = None
        self.is_loaded = False

    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        if self.model_type == "tfidf":
            self.model = TFIDFClassifier()
            self.model.load(model_path)
        elif self.model_type == "rnn":
            if RNNIntentClassifier is None:
                raise ImportError("RNN classifier requires torch. Install with: pip install torch")
            self.model = RNNIntentClassifier()
            self.model.load(model_path)
        elif self.model_type == "semantic":
            self.model = SemanticClassifier()
            self.model.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self.is_loaded = True

    def train_model(self, intents_path: str, save_path: Optional[str] = None) -> dict:
        """Train a new model on intents data."""
        if self.model_type == "tfidf":
            self.model = TFIDFClassifier()
        elif self.model_type == "rnn":
            if RNNIntentClassifier is None:
                raise ImportError("RNN classifier requires torch. Install with: pip install torch")
            self.model = RNNIntentClassifier()
        elif self.model_type == "semantic":
            self.model = SemanticClassifier()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        results = self.model.train(intents_path)
        self.is_loaded = True

        if save_path:
            self.model.save(save_path)

        return results

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify user input and return intent with confidence."""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train_model() first.")

        intent, confidence = self.model.predict(text)

        if confidence < self.confidence_threshold:
            return "fallback", confidence

        return intent, confidence

    def classify_with_fallback(self, text: str) -> Tuple[str, float, bool]:
        """Classify with explicit fallback flag."""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train_model() first.")

        intent, confidence = self.model.predict(text)
        is_fallback = confidence < self.confidence_threshold

        if is_fallback:
            return "fallback", confidence, True

        return intent, confidence, False

    def get_top_intents(self, text: str, top_k: int = 3) -> list:
        """Get top K predicted intents with scores."""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train_model() first.")

        all_scores = self.model.predict_with_scores(text)
        return all_scores[:top_k]


if __name__ == "__main__":
    from config import INTENTS_PATH, TFIDF_MODEL_PATH

    handler = IntentHandler(model_type="tfidf")
    print("Training model...")
    results = handler.train_model(str(INTENTS_PATH), str(TFIDF_MODEL_PATH))
    print(f"Training complete. Accuracy: {results.get('accuracy', 'N/A')}")

    test_queries = [
        "Hi there!",
        "Where is my package?",
        "I want my money back",
        "asdfghjkl random text"
    ]

    print("\nTesting classification:")
    for query in test_queries:
        intent, confidence, is_fallback = handler.classify_with_fallback(query)
        status = "(FALLBACK)" if is_fallback else ""
        print(f"'{query}' -> {intent} ({confidence:.2f}) {status}")
