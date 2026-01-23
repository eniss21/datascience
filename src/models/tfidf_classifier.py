import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing.text_processor import TextProcessor


class TFIDFClassifier:
    """TF-IDF based intent classifier."""

    def __init__(self, max_features: int = 5000):
        # Word-level TF-IDF for semantic matching
        word_vectorizer = TfidfVectorizer(
            max_features=int(max_features * 0.7),
            ngram_range=(1, 3),  
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True
        )
        # Character-level TF-IDF for typo tolerance
        char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 6),  
            max_features=int(max_features * 0.3),
            min_df=1,
            max_df=0.95
        )
        # Combine both vectorizers
        self.vectorizer = FeatureUnion([
            ('word', word_vectorizer),
            ('char', char_vectorizer)
        ])
        self.classifier = LogisticRegression(
            max_iter=3000,
            random_state=42,
            class_weight='balanced',
            C=2.0,  # Increased regularization inverse
            solver='lbfgs'
        )
        self.text_processor = TextProcessor(remove_stopwords=False)
        self.intent_labels: List[str] = []
        self.is_trained = False

    def load_intents(self, intents_path: str) -> Tuple[List[str], List[str]]:
        """Load intents from JSON file."""
        with open(intents_path, 'r') as f:
            data = json.load(f)

        texts = []
        labels = []
        self.intent_labels = []

        for intent in data['intents']:
            tag = intent['tag']
            if tag not in self.intent_labels:
                self.intent_labels.append(tag)

            for pattern in intent['patterns']:
                processed_text = self.text_processor.process(pattern)
                texts.append(processed_text)
                labels.append(tag)

        return texts, labels

    def train(self, intents_path: str, test_size: float = 0.2) -> Dict:
        """Train the classifier on intents data."""
        texts, labels = self.load_intents(intents_path)

        if len(texts) < 10:
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, labels)
            self.is_trained = True
            return {"accuracy": 1.0, "message": "Trained on small dataset without split"}

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train_vec, y_train)
        self.is_trained = True

        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a single text."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        processed_text = self.text_processor.process(text)
        X = self.vectorizer.transform([processed_text])

        intent = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        return intent, confidence

    def predict_with_scores(self, text: str) -> List[Tuple[str, float]]:
        """Predict intent with all class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        processed_text = self.text_processor.process(text)
        X = self.vectorizer.transform([processed_text])

        probabilities = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_

        results = [(cls, float(prob)) for cls, prob in zip(classes, probabilities)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def save(self, path: str):
        """Save model to disk."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'intent_labels': self.intent_labels,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)

    def load(self, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.intent_labels = model_data['intent_labels']
        self.is_trained = model_data['is_trained']


if __name__ == "__main__":
    from config import INTENTS_PATH, TFIDF_MODEL_PATH

    classifier = TFIDFClassifier()
    results = classifier.train(str(INTENTS_PATH))
    print(f"Training accuracy: {results['accuracy']:.4f}")

    classifier.save(str(TFIDF_MODEL_PATH))
    print(f"Model saved to {TFIDF_MODEL_PATH}")

    test_queries = [
        "Hello there!",
        "Where is my order?",
        "I want to return this product",
        "What payment methods do you accept?"
    ]

    for query in test_queries:
        intent, confidence = classifier.predict(query)
        print(f"Query: '{query}' -> Intent: {intent} (confidence: {confidence:.4f})")
