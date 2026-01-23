import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing.text_processor import TextProcessor


class IntentDataset(Dataset):
    """PyTorch Dataset for intent classification."""

    def __init__(self, texts: List[List[int]], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts_padded, labels


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence classification."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch, seq_len, hidden_dim * 2)
        attention_weights = torch.softmax(
            self.attention(lstm_output).squeeze(-1), dim=1
        )  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_dim * 2)

        return context, attention_weights


class IntentClassifier(nn.Module):
    """Improved BiLSTM with attention for intent classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.4,
        num_layers: int = 2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = AttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_output, _ = self.lstm(embedded)
        context, _ = self.attention(lstm_output)
        x = self.relu(self.fc1(self.dropout(context)))
        output = self.fc2(self.dropout(x))
        return output


class RNNIntentClassifier:
    """Wrapper class for training and inference with attention and data augmentation."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.4,
        num_layers: int = 2,
        max_vocab_size: int = 5000,
        max_seq_length: int = 30,
        use_augmentation: bool = True
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.use_augmentation = use_augmentation

        self.text_processor = TextProcessor(remove_stopwords=False)
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.label2idx: Dict[str, int] = {}
        self.idx2label: Dict[int, str] = {}

        self.model: Optional[IntentClassifier] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False

    def build_vocab(self, texts: List[List[str]]):
        """Build vocabulary from tokenized texts."""
        word_counts = Counter()
        for tokens in texts:
            word_counts.update(tokens)

        most_common = word_counts.most_common(self.max_vocab_size - 2)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode_text(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        indices = [self.word2idx.get(token, 1) for token in tokens]
        if len(indices) > self.max_seq_length:
            indices = indices[:self.max_seq_length]
        return indices

    def load_intents(self, intents_path: str) -> Tuple[List[List[str]], List[str]]:
        """Load and tokenize intents from JSON file."""
        with open(intents_path, 'r') as f:
            data = json.load(f)

        tokenized_texts = []
        labels = []

        for intent in data['intents']:
            tag = intent['tag']
            if tag not in self.label2idx:
                idx = len(self.label2idx)
                self.label2idx[tag] = idx
                self.idx2label[idx] = tag

            for pattern in intent['patterns']:
                tokens = self.text_processor.process_to_tokens(pattern)
                tokenized_texts.append(tokens)
                labels.append(tag)

        return tokenized_texts, labels

    def train(
        self,
        intents_path: str,
        epochs: int = 150,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
        early_stop_patience: int = 20
    ) -> Dict:
        """Train the RNN classifier with augmentation and early stopping."""
        tokenized_texts, labels = self.load_intents(intents_path)

        # Apply data augmentation if enabled
        if self.use_augmentation:
            try:
                from src.preprocessing.data_augmentation import DataAugmenter
                augmenter = DataAugmenter()
                augmented_texts = []
                augmented_labels = []

                for tokens, label in zip(tokenized_texts, labels):
                    original_text = ' '.join(tokens)
                    augmented_texts.append(tokens)
                    augmented_labels.append(label)

                    for aug_text in augmenter.augment(original_text, num_augments=2):
                        if aug_text != original_text:
                            aug_tokens = self.text_processor.process_to_tokens(aug_text)
                            augmented_texts.append(aug_tokens)
                            augmented_labels.append(label)

                tokenized_texts = augmented_texts
                labels = augmented_labels
                print(f"Data augmentation: {len(tokenized_texts)} samples")
            except ImportError:
                print("Data augmentation module not available, using original data")

        self.build_vocab(tokenized_texts)

        encoded_texts = [self.encode_text(tokens) for tokens in tokenized_texts]
        encoded_labels = [self.label2idx[label] for label in labels]

        indices = np.random.permutation(len(encoded_texts))
        split_idx = int(len(indices) * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_texts = [encoded_texts[i] for i in train_indices]
        train_labels = [encoded_labels[i] for i in train_indices]
        val_texts = [encoded_texts[i] for i in val_indices]
        val_labels = [encoded_labels[i] for i in val_indices]

        train_dataset = IntentDataset(train_texts, train_labels)
        val_dataset = IntentDataset(val_texts, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        self.model = IntentClassifier(
            vocab_size=len(self.word2idx),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.label2idx),
            dropout=self.dropout,
            num_layers=self.num_layers
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for texts, labels_batch in train_loader:
                texts = texts.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(texts)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for texts, labels_batch in val_loader:
                    texts = texts.to(self.device)
                    labels_batch = labels_batch.to(self.device)

                    outputs = self.model(texts)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()

            val_acc = correct / total if total > 0 else 0
            history["train_loss"].append(train_loss / len(train_loader))
            history["val_loss"].append(val_loss / len(val_loader) if len(val_loader) > 0 else 0)
            history["val_acc"].append(val_acc)

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.is_trained = True
        return {"best_val_accuracy": best_val_acc, "history": history}

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a single text."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        tokens = self.text_processor.process_to_tokens(text)
        encoded = self.encode_text(tokens)

        with torch.no_grad():
            input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            intent = self.idx2label[predicted.item()]
            return intent, confidence.item()

    def predict_with_scores(self, text: str) -> List[Tuple[str, float]]:
        """Predict intent with all class probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        tokens = self.text_processor.process_to_tokens(text)
        encoded = self.encode_text(tokens)

        with torch.no_grad():
            input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            results = [(self.idx2label[i], prob.item()) for i, prob in enumerate(probabilities)]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def save(self, path: str):
        """Save model and vocabulary to disk."""
        checkpoint = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'label2idx': self.label2idx,
            'idx2label': self.idx2label,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'num_layers': self.num_layers,
            'max_vocab_size': self.max_vocab_size,
            'max_seq_length': self.max_seq_length,
            'is_trained': self.is_trained
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model and vocabulary from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        self.word2idx = checkpoint['word2idx']
        self.idx2word = checkpoint['idx2word']
        self.label2idx = checkpoint['label2idx']
        self.idx2label = checkpoint['idx2label']
        self.embed_dim = checkpoint['embed_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.dropout = checkpoint['dropout']
        self.num_layers = checkpoint.get('num_layers', 2)
        self.max_vocab_size = checkpoint['max_vocab_size']
        self.max_seq_length = checkpoint['max_seq_length']
        self.is_trained = checkpoint['is_trained']

        self.model = IntentClassifier(
            vocab_size=len(self.word2idx),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.label2idx),
            dropout=self.dropout,
            num_layers=self.num_layers
        ).to(self.device)

        if checkpoint['model_state_dict']:
            self.model.load_state_dict(checkpoint['model_state_dict'])


if __name__ == "__main__":
    from config import INTENTS_PATH, RNN_MODEL_PATH

    classifier = RNNIntentClassifier()
    results = classifier.train(str(INTENTS_PATH), epochs=100)
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")

    classifier.save(str(RNN_MODEL_PATH))
    print(f"Model saved to {RNN_MODEL_PATH}")

    test_queries = [
        "Hello there!",
        "Where is my order?",
        "I want to return this product",
        "What payment methods do you accept?"
    ]

    for query in test_queries:
        intent, confidence = classifier.predict(query)
        print(f"Query: '{query}' -> Intent: {intent} (confidence: {confidence:.4f})")
