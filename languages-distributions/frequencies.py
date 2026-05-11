import unicodedata
from collections import defaultdict

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def normalize_text(text: str) -> str:
    """
    Normalize text to ASCII: remove diacritics, convert to uppercase.
    é → E, ñ → N, ą → A, ß → SS, etc.
    """
    text = text.upper()
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in normalized if not unicodedata.combining(c))


def unigram_frequencies(text: str) -> dict:
    """
    Compute normalized character frequency distribution over ALPHABET.
    """
    text = normalize_text(text)
    frequencies = {char: 0 for char in ALPHABET}
    for char in text:
        if char in frequencies:
            frequencies[char] += 1
    total = sum(frequencies.values())
    return {char: freq / total for char, freq in frequencies.items()}


def ngram_frequencies(text: str, n: int) -> dict:
    """
    Compute normalized n-gram frequency distribution over ALPHABET.
    Works for bigrams (n=2), trigrams (n=3), etc.
    """
    text = normalize_text(text)
    frequencies = defaultdict(int)
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        if all(c in ALPHABET for c in ngram):
            frequencies[ngram] += 1
    total = sum(frequencies.values())
    return {ngram: freq / total for ngram, freq in frequencies.items()}


def top_n(frequencies: dict, n: int) -> dict:
    """
    Keep only the n most frequent n-grams.
    """
    return dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:n])


def threshold(frequencies: dict, seuil: float) -> dict:
    """
    Remove n-grams below a frequency threshold.
    """
    return {k: v for k, v in frequencies.items() if v > seuil}