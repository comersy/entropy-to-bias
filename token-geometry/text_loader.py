import os

def load_texts(data_dir: str) -> dict:
    """
    Load all .txt files from a directory.
    Returns a dict {language_name: text} where language_name
    is the filename without extension.
    No uppercasing — preserving original case for tokenization.
    """
    texts = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            language = filename.replace('.txt', '')
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                text = text.replace('\n', ' ')
            texts[language] = text
    return texts