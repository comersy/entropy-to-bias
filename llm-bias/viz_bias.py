import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from matplotlib.patches import Patch

family_colors = {
    'French': '#e63946', 'Italian': '#e63946',
    'Spanish': '#e63946', 'Portuguese': '#e63946',
    'English': '#457b9d', 'German': '#457b9d', 'Dutch': '#457b9d',
    'Polish': '#2d6a4f'
}

legend_elements = [
    Patch(facecolor='#e63946', label='Romance'),
    Patch(facecolor='#457b9d', label='Germanic'),
    Patch(facecolor='#2d6a4f', label='Slavic')
]


def plot_embedding_matrix(languages: list, matrix: np.ndarray) -> None:
    """
    Heatmap of pairwise cosine distance matrix in embedding space.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        xticklabels=languages,
        yticklabels=languages,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        linewidths=0.3,
        ax=ax,
        cbar=False
    )
    ax.set_title('Pairwise cosine distance matrix — XLM-R embedding space', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    plt.savefig('../data/results/embedding_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_embedding_mds(languages: list, matrix: np.ndarray) -> None:
    """
    MDS projection of the embedding space distance matrix.
    """
    matrix_sym = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix_sym, 0)

    reducer = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embedding = reducer.fit_transform(matrix_sym)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, lang in enumerate(languages):
        color = family_colors.get(lang, 'gray')
        ax.scatter(embedding[i, 0], embedding[i, 1], s=150, color=color, zorder=3)
        ax.annotate(lang, (embedding[i, 0], embedding[i, 1]),
                   xytext=(6, 4), textcoords='offset points', fontsize=10)

    ax.set_title('MDS — XLM-R embedding space', fontsize=12)
    ax.legend(handles=legend_elements, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('../data/results/embedding_mds.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_mantel_distribution(observed_r: float, permuted_rs: np.ndarray, 
                              name1: str, name2: str) -> None:
    """
    Plot the null distribution of Mantel correlations vs observed value.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(permuted_rs, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(observed_r, color='#e63946', linewidth=2, label=f'Observed r = {observed_r:.3f}')
    ax.set_xlabel('Correlation coefficient')
    ax.set_ylabel('Count')
    ax.set_title(f'Mantel test — {name1} vs {name2}')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'../data/results/mantel_{name1.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_mantel_summary(results: dict) -> None:

    names = list(results.keys())
    rs = [results[n][0] for n in names]
    pvals = [results[n][1] for n in names]
    colors = ['#2d6a4f' if p < 0.05 else '#e63946' for p in pvals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, rs, color=colors, edgecolor='white')
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.set_ylabel('Mantel r')
    ax.set_title('Mantel correlations — statistical distances vs embedding distances')
    ax.set_ylim(-0.2, 1.1)

    for bar, val, p in zip(bars, rs, pvals):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}\n{sig}', ha='center', fontsize=9)

    ax.tick_params(axis='x', rotation=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('../data/results/mantel_summary.png', dpi=150, bbox_inches='tight')
    plt.show()