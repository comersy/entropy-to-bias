import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from metrics import entropy

def plot_entropy(lang_distributions: dict) -> None:
    """
    Bar chart of Shannon entropy per language.
    Higher entropy = more uniform distribution = less predictable.
    """
    languages = list(lang_distributions.keys())
    entropies = [entropy(dist) for dist in lang_distributions.values()]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(languages, entropies, color='steelblue', edgecolor='white')
    plt.ylabel('Shannon entropy (bits)')
    plt.title('Entropy per language — character level')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_distance_matrix(languages: list, matrix: np.ndarray, metric: str = 'hellinger') -> None:
    """
    Heatmap of pairwise distance matrix between languages.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=languages,
        yticklabels=languages,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        linewidths=0.5
    )
    plt.title(f'Pairwise distance matrix — {metric}')
    plt.tight_layout()
    plt.show()


def plot_frequency_distributions(lang_distributions: dict) -> None:
    """
    Side by side bar charts of character frequency distributions.
    """
    n = len(lang_distributions)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))

    for ax, (lang, dist) in zip(axes, lang_distributions.items()):
        chars = list(dist.keys())
        freqs = list(dist.values())
        ax.bar(chars, freqs, color='steelblue', edgecolor='white')
        ax.set_title(lang)
        ax.set_ylabel('Frequency')
        ax.set_ylim(0, max(freqs) * 1.2)

    plt.tight_layout()
    plt.show()


def plot_umap(languages: list, matrix: np.ndarray, metric: str = 'hellinger') -> None:
    """
    UMAP projection of the pairwise distance matrix.
    Each point is a language — nearby points have similar character distributions.
    """
    reducer = UMAP(n_components=2, metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=100, color='steelblue')

    for i, lang in enumerate(languages):
        plt.annotate(
            lang,
            (embedding[i, 0], embedding[i, 1]),
            fontsize=11,
            xytext=(8, 4),
            textcoords='offset points'
        )

    plt.title(f'UMAP projection of language distances — {metric}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()