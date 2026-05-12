# token_frequencies.py
from collections import defaultdict
from transformers import AutoTokenizer

MODEL_NAME = "xlm-roberta-base"

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def token_frequencies(text: str, tokenizer) -> dict:
    """
    Compute normalized token frequency distribution for a given text.
    Uses XLM-R tokenizer (BPE). Returns a dict {token_id: frequency}.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    frequencies = defaultdict(int)
    for token in tokens:
        frequencies[token] += 1
    total = sum(frequencies.values())
    return {token: freq / total for token, freq in frequencies.items()}


def token_frequencies_readable(text: str, tokenizer) -> dict:
    """
    Same as token_frequencies but returns {token_string: frequency}
    instead of {token_id: frequency}. Easier to inspect.
    """
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(text, add_special_tokens=False)
    )
    frequencies = defaultdict(int)
    for token in tokens:
        frequencies[token] += 1
    total = sum(frequencies.values())
    return {token: freq / total for token, freq in frequencies.items()}