import numpy as np
from metrics import kl_divergence, hellinger, bhattacharyya

METRICS = {
    'kl_divergence': kl_divergence,
    'hellinger': hellinger,
    'bhattacharyya': bhattacharyya
}

def compute_distance_matrix(lang_distributions: dict, metric: str = 'hellinger') -> tuple:
    """
    Compute pairwise distance matrix between language distributions.
    
    Args:
        lang_distributions: dict {language_name: frequency_distribution}
        metric: one of 'kl_divergence', 'hellinger', 'bhattacharyya'
    
    Returns:
        languages: list of language names
        matrix: np.ndarray of shape (n_languages, n_languages)
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(METRICS.keys())}")
    
    metric_fn = METRICS[metric]
    languages = list(lang_distributions.keys())
    n = len(languages)
    matrix = np.zeros((n, n))

    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i != j:
                matrix[i, j] = metric_fn(
                    lang_distributions[lang1],
                    lang_distributions[lang2]
                )
    return languages, matrix


def save_matrix(matrix: np.ndarray, languages: list, path: str) -> None:
    """
    Save distance matrix and language labels to disk.
    """
    np.save(path + '_matrix.npy', matrix)
    np.save(path + '_languages.npy', np.array(languages))


def load_matrix(path: str) -> tuple:
    """
    Load distance matrix and language labels from disk.
    """
    matrix = np.load(path + '_matrix.npy')
    languages = np.load(path + '_languages.npy').tolist()
    return languages, matrix