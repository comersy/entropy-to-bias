# entropy-to-bias

> Can the statistical signature of a language predict how a large language model treats it?

This project investigates the link between **information-theoretic properties of languages** and **multilingual LLM bias**. It is structured as a progression: we start from raw text, build statistical representations of languages using entropy and divergence metrics, and work our way up to the geometry of embedding spaces in XLM-R.

---

## The idea

A language is, at its core, a probability distribution over symbols. French and Spanish distribute their characters differently than Polish or Dutch. These differences are measurable — using Shannon entropy, KL divergence, Hellinger distance — and they are not arbitrary: they reflect deep structural properties of each language (morphology, redundancy, compressibility).

The hypothesis: **if a multilingual LLM encodes languages based on their statistical properties, then information-theoretic distances between languages should predict distances in the model's embedding space** — and by extension, the model's differential treatment of languages.

We test this hypothesis in three steps.

---

## Act 1 — Languages as distributions

We treat each language as a probability distribution over characters. Using one reference text per language (literary works from Project Gutenberg), we compute character frequency distributions and measure pairwise distances using three metrics: KL divergence, Hellinger distance, and Bhattacharyya distance.

The result: **language family structure emerges from raw statistics alone**, with no linguistic knowledge injected.

<p align="center">
  <img src="data/results/char_mds_kl_divergence_1.png" width="500"/>
</p>

Romance languages (French, Italian, Spanish, Portuguese) cluster tightly. Germanic languages (English, German, Dutch) form a separate group. Polish stands alone — the only Slavic language in our dataset, and consistently the most distant from all others.

We then benchmark language detection: given a short extract of ~200 characters, can we identify its language by finding the closest reference distribution? We test all combinations of metric × n-gram order (unigrams, bigrams, trigrams) on 50 random extracts per language, and find the best-performing combination.

`notebooks/01_languages_as_distributions.ipynb`

---

## Act 2 — From characters to tokens

A language model does not see characters. It sees **BPE tokens** — subword units produced by a tokenizer trained on a large multilingual corpus. The tokenizer itself is a statistical object: it was built by merging frequent character pairs, and its vocabulary reflects the distribution of subwords across all the languages it was trained on.

We repeat the entire pipeline from Act 1 at the token level, using the XLM-R tokenizer, and introduce a new metric: **tokenizer fertility** — the average number of tokens per word.

<p align="center">
  <img src="data/results/fertility.png" width="600"/>
</p>

Fertility is not uniform across languages. Some languages are tokenized efficiently (most words map to a single token); others are systematically fragmented. This asymmetry is structural — it is baked into the tokenizer before any model training happens, and it is already a form of bias.

We then recompute distance matrices and MDS projections at the token level, and compare them to the character-level results. The language family structure largely survives tokenization, but the geometry shifts — the tokenizer introduces its own distortions on top of the linguistic signal.

<p align="center">
  <img src="data/results/token_mds_hellinger.png" width="500"/>
</p>

`notebooks/02_from_characters_to_tokens.ipynb`

---

## Act 3 — Predicting LLM bias

We now extract language representations from XLM-R using **parallel sentences** from the opus-100 corpus (Helsinki-NLP). By holding content constant across languages, any variation in embedding space must come from the language itself.

For each language, we pass parallel sentences through XLM-R, extract sentence embeddings via mean pooling, and average them into a single language vector. We then compute pairwise cosine distances between these vectors.

<p align="center">
  <img src="data/results/embedding_mds.png" width="500"/>
</p>

Language family structure is visible in XLM-R's embedding space — Romance languages cluster together, Germanic languages occupy a separate region, Polish remains isolated. The absolute distances are small (order of 0.001–0.003), reflecting XLM-R's design as a shared multilingual space, but the structure is real.

We then run **Mantel tests** — the standard method for correlating two distance matrices — between our statistical distance matrices (from Acts 1 and 2) and the embedding distance matrix.

<p align="center">
  <img src="data/results/mantel_summary.png" width="600"/>
</p>

**Results:**
- Character bigrams (Hellinger) vs XLM-R embeddings: **r = 0.557, p = 0.007**
- Token level (Hellinger) vs XLM-R embeddings: **r = 0.515, p = 0.006**

Both correlations are significant at p < 0.01. The statistical structure of languages — measurable from raw text using nothing but frequency distributions — partially predicts the geometry that XLM-R has learned in its embedding space.

`notebooks/03_predicting_llm_bias.ipynb`

---

## What this means

The geometry a multilingual model learns is not arbitrary. Languages that are statistically close tend to be close in the model's internal representation, and languages that are statistically distant tend to be distant there too. This suggests that a part of the model's differential treatment of languages — its bias — is structurally driven by the distributional properties of the languages themselves, properties that are measurable from raw text before any model is involved.

This does not mean statistical distance is the only driver. Training data volume, tokenizer design, and fine-tuning choices all play a role. And correlation is not causation. But the signal is there.

---

## Limitations and next steps

- **8 languages is a small sample.** 28 pairs give the Mantel test limited statistical power. Extending to 30–50 typologically diverse languages would substantially strengthen or challenge these findings.
- **Single model, single layer.** Different models (mBERT, BLOOM, LLaMA) and different layers may tell a different story.
- **opus-100 is not perfectly parallel.** English source sentences differ across language pairs, introducing noise into the embedding comparison.
- **Partial Mantel tests** controlling for training data volume and language family membership would isolate the specific contribution of statistical distance.

---

## Languages

French, English, German, Spanish, Italian, Dutch, Portuguese, Polish — spanning Romance, Germanic, and Slavic families, chosen to be close enough to make statistical differences subtle, far enough to expect meaningful variation.

---

## Repository structure

```
entropy-to-bias/
│
├── data/
│   ├── Books/          ← reference texts (Project Gutenberg)
│   └── results/        ← saved figures
│
├── languages-distributions/
│   ├── text_loader.py
│   ├── frequencies.py
│   ├── metrics.py
│   ├── distance_matrix.py
│   ├── detection.py
│   └── viz_characters.py
│
├── token-geometry/
│   ├── text_loader.py
│   ├── token_frequencies.py
│   ├── fertility.py
│   └── viz_tokens.py
│
├── llm-bias/
│   ├── flores_loader.py
│   ├── embeddings.py
│   ├── mantel.py
│   └── viz_bias.py
│
└── notebooks/
    ├── 01_languages_as_distributions.ipynb
    ├── 02_from_characters_to_tokens.ipynb
    └── 03_predicting_llm_bias.ipynb
```

---

## Setup

```bash
git clone https://github.com/comersy/entropy-to-bias
cd entropy-to-bias
pip install -r requirements.txt
```


