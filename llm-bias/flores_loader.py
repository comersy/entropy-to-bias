from datasets import load_dataset

OPUS_LANGUAGES = {
    'French':     'en-fr',
    'German':     'de-en',
    'Spanish':    'en-es',
    'Italian':    'en-it',
    'Portuguese': 'en-pt',
    'Dutch':      'en-nl',
    'Polish':     'en-pl'
}

def load_flores(split: str = 'test') -> dict:
    """
    Load parallel sentences for our 8 languages using opus-100.
    Returns a dict {language: list of sentences}.
    """
    print("Loading opus-100...")
    texts = {'English': []}

    for language, pair in OPUS_LANGUAGES.items():
        dataset = load_dataset('Helsinki-NLP/opus-100', pair, split='test')
        src, tgt = pair.split('-')
        en_key = 'en'
        lang_key = tgt if src == 'en' else src
        
        texts['English'] += [row['translation'][en_key] for row in dataset]
        texts[language] = [row['translation'][lang_key] for row in dataset]
        print(f"  {language}: {len(texts[language])} sentences loaded")

    texts['English'] = texts['English'][:len(texts['French'])]
    return texts