# fertility.py
from token_frequencies import load_tokenizer

def fertility(text: str, tokenizer) -> float:
    words = text.split()
    if not words:
        return 0.0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens) / len(words)


def fertility_per_language(texts: dict, tokenizer) -> dict:
    return {lang: fertility(text, tokenizer) for lang, text in texts.items()}