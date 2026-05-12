import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "xlm-roberta-base"

def load_model():
    """
    Load XLM-R tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def get_sentence_embedding(sentence: str, tokenizer, model, layer: int = -1) -> np.ndarray:
    """
    Extract sentence embedding from XLM-R.
    Uses mean pooling over token embeddings at the specified layer.
    layer=-1 means last layer, layer=-6 means 6th layer from the end.
    """
    inputs = tokenizer(
        sentence,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Hidden states: tuple of (n_layers + 1) tensors of shape (batch, seq_len, hidden_size)
    hidden_state = outputs.hidden_states[layer]

    # Mean pooling over token dimension (excluding padding)
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    embedding = (hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    return embedding.squeeze().numpy()


def get_language_embedding(sentences: list, tokenizer, model, layer: int = -1) -> np.ndarray:
    """
    Compute mean embedding for a language from a list of parallel sentences.
    Returns a single vector representing the language in embedding space.
    """
    embeddings = []
    for sentence in tqdm(sentences, leave=False):
        emb = get_sentence_embedding(sentence, tokenizer, model, layer=layer)
        embeddings.append(emb)
    return np.mean(embeddings, axis=0)


def get_all_language_embeddings(flores_texts: dict, tokenizer, model, layer: int = -1) -> dict:
    """
    Compute mean embeddings for all languages.
    Returns dict {language: embedding_vector}.
    """
    embeddings = {}
    for lang, sentences in flores_texts.items():
        print(f"Computing embedding for {lang}...")
        embeddings[lang] = get_language_embedding(sentences, tokenizer, model, layer=layer)
    return embeddings


def cosine_distance_matrix(embeddings: dict) -> tuple:
    """
    Compute pairwise cosine distance matrix between language embeddings.
    Cosine distance = 1 - cosine similarity.
    Returns (languages, matrix).
    """
    languages = list(embeddings.keys())
    n = len(languages)
    matrix = np.zeros((n, n))

    vecs = np.array([embeddings[l] for l in languages])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_normalized = vecs / norms

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = 1 - np.dot(vecs_normalized[i], vecs_normalized[j])

    return languages, matrix