# visualization.py
import matplotlib.pyplot as plt

def plot_fertility(fertility_scores: dict) -> None:
    langs = sorted(fertility_scores, key=fertility_scores.get)
    values = [fertility_scores[l] for l in langs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(langs, values, color='steelblue', edgecolor='white')
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Tokens per word')
    ax.set_title('Tokenizer fertility per language — XLM-R')

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()