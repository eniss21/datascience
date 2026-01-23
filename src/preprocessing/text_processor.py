import re
import string
from typing import List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class TextProcessor:
    """Text preprocessing for NLP chatbot."""

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_data()

        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()

    def _download_nltk_data(self):
        """Download required NLTK data."""
        nltk_packages = ['punkt', 'wordnet', 'stopwords', 'punkt_tab']
        for package in nltk_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
            except LookupError:
                nltk.download(package, quiet=True)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        return tokens

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [token for token in tokens if token not in self.stop_words]

    def process_for_semantic(self, text: str) -> str:
        """Light preprocessing for semantic models - preserves natural text structure."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self, text: str) -> str:
        """Full preprocessing pipeline. Returns processed string."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        return ' '.join(tokens)

    def process_to_tokens(self, text: str) -> List[str]:
        """Full preprocessing pipeline. Returns list of tokens."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        return tokens

    def batch_process(self, texts: List[str]) -> List[str]:
        """Process multiple texts."""
        return [self.process(text) for text in texts]
