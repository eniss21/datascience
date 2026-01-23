"""Data augmentation for NLP training data."""

import random
from typing import List
import nltk
from nltk.corpus import wordnet


class DataAugmenter:
    """Data augmentation techniques for text classification."""

    def __init__(self):
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Download required NLTK data."""
        for package in ['wordnet', 'averaged_perceptron_tagger', 'omw-1.4']:
            try:
                nltk.data.find(f'corpora/{package}')
            except LookupError:
                nltk.download(package, quiet=True)

    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms."""
        words = text.split()
        if len(words) == 0:
            return text

        new_words = words.copy()
        random_word_indices = list(range(len(words)))
        random.shuffle(random_word_indices)

        replacements = 0
        for idx in random_word_indices:
            word = words[idx]
            synonyms = self._get_synonyms(word)
            if synonyms:
                new_words[idx] = random.choice(synonyms)
                replacements += 1
                if replacements >= n:
                    break

        return ' '.join(new_words)

    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random synonyms at random positions."""
        words = text.split()
        if len(words) == 0:
            return text

        new_words = words.copy()
        for _ in range(n):
            word = random.choice(words)
            synonyms = self._get_synonyms(word)
            if synonyms:
                insert_pos = random.randint(0, len(new_words))
                new_words.insert(insert_pos, random.choice(synonyms))

        return ' '.join(new_words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Delete words with probability p."""
        words = text.split()
        if len(words) <= 1:
            return text

        new_words = [w for w in words if random.random() > p]
        if not new_words:
            return random.choice(words)

        return ' '.join(new_words)

    def random_swap(self, text: str, n: int = 1) -> str:
        """Swap n pairs of words."""
        words = text.split()
        if len(words) < 2:
            return text

        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return ' '.join(new_words)

    def augment(self, text: str, num_augments: int = 4) -> List[str]:
        """Generate multiple augmented versions of text."""
        augmented = [text]  # Include original

        methods = [
            lambda t: self.synonym_replacement(t, n=1),
            lambda t: self.random_insertion(t, n=1),
            lambda t: self.random_deletion(t, p=0.1),
            lambda t: self.random_swap(t, n=1),
        ]

        for _ in range(num_augments):
            method = random.choice(methods)
            aug_text = method(text)
            if aug_text and aug_text != text:
                augmented.append(aug_text)

        return augmented


def augment_intents_data(texts: List[str], labels: List[str], num_augments: int = 2) -> tuple:
    """Augment training data with additional examples."""
    augmenter = DataAugmenter()
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        # Add original
        augmented_texts.append(text)
        augmented_labels.append(label)

        # Add augmented versions
        for aug_text in augmenter.augment(text, num_augments=num_augments):
            if aug_text != text:
                augmented_texts.append(aug_text)
                augmented_labels.append(label)

    return augmented_texts, augmented_labels
