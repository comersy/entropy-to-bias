# detection.py
from frequencies import unigram_frequencies, ngram_frequencies
from metrics import kl_divergence, hellinger, bhattacharyya

METRICS = {
    'kl_divergence': kl_divergence,
    'hellinger': hellinger,
    'bhattacharyya': bhattacharyya
}

def detect_language(text: str, lang_distributions: dict, metric: str = 'hellinger', n: int = 1) -> tuple:
    """
    Detect the language of a text by finding the closest language distribution.

    Args:
        text: input text to identify
        lang_distributions: dict {language_name: frequency_distribution}
        metric: one of 'kl_divergence', 'hellinger', 'bhattacharyya'
        n: n-gram order (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        detected_language: name of the detected language
        scores: dict {language: distance} for all languages
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(METRICS.keys())}")

    metric_fn = METRICS[metric]

    if n == 1:
        text_distribution = unigram_frequencies(text)
    else:
        text_distribution = ngram_frequencies(text, n)

    scores = {}
    for lang, distribution in lang_distributions.items():
        scores[lang] = metric_fn(text_distribution, distribution)

    detected_language = min(scores, key=scores.get)
    return detected_language, scores