"""
Semantic Intent Classifier using Sentence Embeddings.
This classifier understands meaning, not just keywords.
"call agent" will match "speak with representative", "talk to human", etc.
"""

import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Literal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing.text_processor import TextProcessor


class SemanticClassifier:
    """Intent classifier using semantic similarity with sentence embeddings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_preprocessing: bool = False,
        prediction_method: Literal["patterns", "intent_average", "hybrid"] = "patterns"
    ):
        """
        Initialize semantic classifier.
        
        Args:
            model_name: Sentence transformer model name
            use_preprocessing: Whether to preprocess text before encoding
            prediction_method: 
                - "patterns": Compare against all individual patterns (most accurate, slower)
                - "intent_average": Compare against average intent embeddings (faster)
                - "hybrid": Use both methods and combine scores
        """
        self.model_name = model_name
        self.model = None
        self.use_preprocessing = use_preprocessing
        self.prediction_method = prediction_method
        
        self.text_processor = TextProcessor(remove_stopwords=False) if use_preprocessing else None
        
        self.intent_embeddings: Dict[str, np.ndarray] = {}  # intent -> average embedding
        self.pattern_embeddings: List[Tuple[str, np.ndarray]] = []  # (intent, embedding) pairs
        self.intent_labels: List[str] = []
        self.intents_data: Dict = {}
        self.is_trained = False

    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                print(f"Loaded sentence transformer model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "Please install sentence-transformers: pip install sentence-transformers"
                )

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text if enabled."""
        if self.use_preprocessing and self.text_processor:
            return self.text_processor.process(text)
        return text

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        self._load_model()
        processed_text = self._preprocess_text(text)
        return self.model.encode(processed_text, normalize_embeddings=True, show_progress_bar=False)

    def _get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for multiple texts with batching."""
        self._load_model()
        processed_texts = [self._preprocess_text(text) for text in texts]
        return self.model.encode(
            processed_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )

    def load_intents(self, intents_path: str) -> Tuple[List[str], List[str]]:
        """Load intents from JSON file."""
        with open(intents_path, 'r') as f:
            data = json.load(f)

        texts = []
        labels = []
        self.intent_labels = []
        self.intents_data = {intent['tag']: intent for intent in data['intents']}

        for intent in data['intents']:
            tag = intent['tag']
            if tag not in self.intent_labels:
                self.intent_labels.append(tag)

            for pattern in intent['patterns']:
                texts.append(pattern)
                labels.append(tag)

        return texts, labels

    def train(self, intents_path: str, test_size: float = 0.2) -> Dict:
        """
        Train by computing embeddings for all patterns.
        
        Args:
            intents_path: Path to intents JSON file
            test_size: Fraction of data to use for testing (0.0 to skip evaluation)
        
        Returns:
            Dictionary with training results and metrics
        """
        print("Loading intents...")
        texts, labels = self.load_intents(intents_path)
        
        print(f"Computing embeddings for {len(texts)} patterns...")
        self.pattern_embeddings = []
        self.intent_embeddings = {}
        
        # Group patterns by intent for efficient batch processing
        intent_patterns: Dict[str, List[str]] = {}
        for text, label in zip(texts, labels):
            if label not in intent_patterns:
                intent_patterns[label] = []
            intent_patterns[label].append(text)

        # Process each intent
        for tag, patterns in intent_patterns.items():
            if not patterns:  # Skip fallback with no patterns
                continue

            print(f"  Processing intent: {tag} ({len(patterns)} patterns)")
            # Get embeddings for all patterns in this intent
            embeddings = self._get_embeddings(patterns)

            # Store each pattern's embedding
            for pattern, emb in zip(patterns, embeddings):
                self.pattern_embeddings.append((tag, emb))

            # Store average embedding for the intent
            self.intent_embeddings[tag] = np.mean(embeddings, axis=0)

        self.is_trained = True
        
        result = {
            "status": "trained",
            "num_intents": len(self.intent_embeddings),
            "num_patterns": len(self.pattern_embeddings)
        }
        
        # Evaluate on test set if requested
        if test_size > 0 and len(texts) >= 10:
            print(f"\nEvaluating on test set (test_size={test_size})...")
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            correct = 0
            total = len(X_test)
            predictions = []
            
            for text, true_label in zip(X_test, y_test):
                predicted_intent, _ = self.predict(text)
                predictions.append(predicted_intent)
                if predicted_intent == true_label:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0.0
            result["accuracy"] = accuracy
            result["test_samples"] = total
            
            # Generate classification report
            try:
                report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
                result["classification_report"] = report
            except Exception as e:
                print(f"Warning: Could not generate classification report: {e}")
        
        return result

    def predict(
        self,
        text: str,
        method: Optional[Literal["patterns", "intent_average", "hybrid"]] = None
    ) -> Tuple[str, float]:
        """
        Predict intent using semantic similarity.
        
        Args:
            text: Input text to classify
            method: Override default prediction method
        
        Returns:
            Tuple of (intent, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        method = method or self.prediction_method
        text_embedding = self._get_embedding(text)

        if method == "patterns":
            return self._predict_patterns(text_embedding)
        elif method == "intent_average":
            return self._predict_intent_average(text_embedding)
        elif method == "hybrid":
            return self._predict_hybrid(text_embedding)
        else:
            raise ValueError(f"Unknown prediction method: {method}")

    def _predict_patterns(self, text_embedding: np.ndarray) -> Tuple[str, float]:
        """Predict by comparing against all individual patterns (most accurate)."""
        best_intent = "fallback"
        best_score = 0.0

        for intent, pattern_emb in self.pattern_embeddings:
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                pattern_emb.reshape(1, -1)
            )[0][0]

            if similarity > best_score:
                best_score = similarity
                best_intent = intent

        return best_intent, float(best_score)

    def _predict_intent_average(self, text_embedding: np.ndarray) -> Tuple[str, float]:
        """Predict by comparing against average intent embeddings (faster)."""
        best_intent = "fallback"
        best_score = 0.0

        for intent, intent_emb in self.intent_embeddings.items():
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                intent_emb.reshape(1, -1)
            )[0][0]

            if similarity > best_score:
                best_score = similarity
                best_intent = intent

        return best_intent, float(best_score)

    def _predict_hybrid(self, text_embedding: np.ndarray) -> Tuple[str, float]:
        """Predict using both methods and combine scores."""
        # Get scores from both methods
        intent_pattern_scores = {}
        intent_avg_scores = {}
        
        # Pattern-based scores
        for intent, pattern_emb in self.pattern_embeddings:
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                pattern_emb.reshape(1, -1)
            )[0][0]
            
            if intent not in intent_pattern_scores or similarity > intent_pattern_scores[intent]:
                intent_pattern_scores[intent] = similarity
        
        # Intent average scores
        for intent, intent_emb in self.intent_embeddings.items():
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                intent_emb.reshape(1, -1)
            )[0][0]
            intent_avg_scores[intent] = similarity
        
        # Combine scores (weighted average: 70% patterns, 30% average)
        combined_scores = {}
        all_intents = set(intent_pattern_scores.keys()) | set(intent_avg_scores.keys())
        
        for intent in all_intents:
            pattern_score = intent_pattern_scores.get(intent, 0.0)
            avg_score = intent_avg_scores.get(intent, 0.0)
            combined_scores[intent] = 0.7 * pattern_score + 0.3 * avg_score
        
        if not combined_scores:
            return "fallback", 0.0
        
        best_intent = max(combined_scores, key=combined_scores.get)
        best_score = combined_scores[best_intent]
        
        return best_intent, float(best_score)

    def predict_with_scores(
        self,
        text: str,
        top_k: int = 5,
        method: Optional[Literal["patterns", "intent_average", "hybrid"]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top K intents with similarity scores.
        
        Args:
            text: Input text to classify
            top_k: Number of top intents to return
            method: Override default prediction method
        
        Returns:
            List of (intent, score) tuples sorted by score descending
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        method = method or self.prediction_method
        text_embedding = self._get_embedding(text)

        if method == "patterns":
            # Get best pattern match for each intent
            intent_scores = {}
            for intent, pattern_emb in self.pattern_embeddings:
                similarity = cosine_similarity(
                    text_embedding.reshape(1, -1),
                    pattern_emb.reshape(1, -1)
                )[0][0]
                
                if intent not in intent_scores or similarity > intent_scores[intent]:
                    intent_scores[intent] = similarity
            
            scores = [(intent, float(score)) for intent, score in intent_scores.items()]
        elif method == "intent_average":
            # Calculate similarity to each intent's average embedding
            scores = []
            for intent, intent_emb in self.intent_embeddings.items():
                similarity = cosine_similarity(
                    text_embedding.reshape(1, -1),
                    intent_emb.reshape(1, -1)
                )[0][0]
                scores.append((intent, float(similarity)))
        else:  # hybrid
            # Use hybrid method
            intent_pattern_scores = {}
            intent_avg_scores = {}
            
            for intent, pattern_emb in self.pattern_embeddings:
                similarity = cosine_similarity(
                    text_embedding.reshape(1, -1),
                    pattern_emb.reshape(1, -1)
                )[0][0]
                if intent not in intent_pattern_scores or similarity > intent_pattern_scores[intent]:
                    intent_pattern_scores[intent] = similarity
            
            for intent, intent_emb in self.intent_embeddings.items():
                similarity = cosine_similarity(
                    text_embedding.reshape(1, -1),
                    intent_emb.reshape(1, -1)
                )[0][0]
                intent_avg_scores[intent] = similarity
            
            all_intents = set(intent_pattern_scores.keys()) | set(intent_avg_scores.keys())
            scores = []
            for intent in all_intents:
                pattern_score = intent_pattern_scores.get(intent, 0.0)
                avg_score = intent_avg_scores.get(intent, 0.0)
                combined = 0.7 * pattern_score + 0.3 * avg_score
                scores.append((intent, float(combined)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str):
        """Save model data to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        # Convert numpy arrays to lists for pickle compatibility
        data = {
            'model_name': self.model_name,
            'use_preprocessing': self.use_preprocessing,
            'prediction_method': self.prediction_method,
            'intent_embeddings': {k: v.tolist() for k, v in self.intent_embeddings.items()},
            'pattern_embeddings': [(intent, emb.tolist()) for intent, emb in self.pattern_embeddings],
            'intent_labels': self.intent_labels,
            'intents_data': self.intents_data,
            'is_trained': self.is_trained
        }
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model data from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model_name = data['model_name']
        self.use_preprocessing = data.get('use_preprocessing', True)
        self.prediction_method = data.get('prediction_method', 'patterns')
        
        # Convert lists back to numpy arrays
        self.intent_embeddings = {
            k: np.array(v) for k, v in data['intent_embeddings'].items()
        }
        self.pattern_embeddings = [
            (intent, np.array(emb)) for intent, emb in data['pattern_embeddings']
        ]
        
        self.intent_labels = data.get('intent_labels', [])
        self.intents_data = data.get('intents_data', {})
        self.is_trained = data['is_trained']
        
        # Reinitialize text processor if needed
        if self.use_preprocessing:
            self.text_processor = TextProcessor(remove_stopwords=False)
        
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    from config import INTENTS_PATH, MODELS_DIR

    SEMANTIC_MODEL_PATH = MODELS_DIR / "semantic_model.pkl"

    print("=" * 70)
    print("Semantic Intent Classifier - Training and Testing")
    print("=" * 70)
    
    # Initialize classifier
    classifier = SemanticClassifier(
        model_name="all-MiniLM-L6-v2",
        use_preprocessing=False,  # Disabled - sentence transformers work better with natural text
        prediction_method="patterns"  # Most accurate
    )
    
    # Train
    print("\nTraining semantic classifier...")
    results = classifier.train(str(INTENTS_PATH), test_size=0.2)
    
    print(f"\nTraining Results:")
    print(f"  Status: {results['status']}")
    print(f"  Number of intents: {results['num_intents']}")
    print(f"  Number of patterns: {results['num_patterns']}")
    if 'accuracy' in results:
        print(f"  Test Accuracy: {results['accuracy']:.4f}")
        print(f"  Test Samples: {results['test_samples']}")
    
    # Save model
    classifier.save(str(SEMANTIC_MODEL_PATH))
    
    # Test with phrases NOT in training data (semantic understanding)
    print("\n" + "=" * 70)
    print("Testing Semantic Understanding (phrases NOT in training data)")
    print("=" * 70)
    
    test_queries = [
        ("call agent", "contact_human"),
        ("can i speak with an agent", "contact_human"),
        ("i need to talk to someone", "contact_human"),
        ("connect me to support", "contact_human"),
        ("hey", "greeting"),
        ("yo whats up", "greeting"),
        ("track my purchase", "order_status"),
        ("where did my stuff go", "order_status"),
        ("give me my money back", "refund_status"),
        ("this is broken i want refund", "return_request"),
        ("i want to revoke my purchase", "order_cancel"),
        ("when will my package arrive", "order_status"),
        ("how do i pay", "payment_methods"),
        ("what cards do you take", "payment_methods"),
    ]

    print("\nPrediction Results:")
    print("-" * 70)
    correct = 0
    total = len(test_queries)
    
    for query, expected_intent in test_queries:
        intent, confidence = classifier.predict(query)
        top_intents = classifier.predict_with_scores(query, top_k=3)
        
        match = "✓" if intent == expected_intent else "✗"
        if intent == expected_intent:
            correct += 1
        
        print(f"\n{match} Query: '{query}'")
        print(f"   Expected: {expected_intent}")
        print(f"   Predicted: {intent} (confidence: {confidence:.2%})")
        print(f"   Top 3: {', '.join([f'{i[0]}({i[1]:.2%})' for i in top_intents])}")
    
    print("\n" + "=" * 70)
    print(f"Semantic Understanding Accuracy: {correct}/{total} ({correct/total:.2%})")
    print("=" * 70)
    
    # Test loading
    print("\nTesting model loading...")
    loaded_classifier = SemanticClassifier()
    loaded_classifier.load(str(SEMANTIC_MODEL_PATH))
    
    test_query = "i need human help"
    intent, confidence = loaded_classifier.predict(test_query)
    print(f"Loaded model test: '{test_query}' -> {intent} ({confidence:.2%})")
    print("Model loading successful!")
