# build_vocab_and_embeddings.py

import os
import json
import numpy as np

from gensim.models import KeyedVectors

# Step 1: Build vocab from your corpus
def build_vocab_from_texts(texts, min_freq=1):
    from collections import Counter
    vocab_counts = Counter()
    for text in texts:
        tokens = text.split()   # basic tokenizer, better: use spaCy or NLTK word_tokenize
        vocab_counts.update(tokens)
    # Start with reserved tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in vocab_counts.items():
        if count >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab

# Step 2: Build embedding matrix for vocab using a pretrained embedding
def create_embedding_matrix(vocab, embedding_model, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    for word, idx in vocab.items():
        if word in embedding_model:
            embedding_matrix[idx] = embedding_model[word]
        else:
            # Random vector for OOV/UNK, or you can keep zeros
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

if __name__ == "__main__":
    # Gather all cleaned texts
    corpus = []
    doc_folder = "processed_documents"
    for fname in os.listdir(doc_folder):
        if fname.endswith("_cleaned.txt"):
            with open(os.path.join(doc_folder, fname), encoding="utf-8") as f:
                corpus.append(f.read())

    # Build vocab
    vocab = build_vocab_from_texts(corpus, min_freq=1)
    with open("vocab.json", "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"Saved vocab.json ({len(vocab)} tokens)")

    # Load a pretrained embedding model, e.g., GloVe (txt) or Word2Vec (bin)
    # For GloVe example:
    # Download "glove.6B.100d.txt" from https://nlp.stanford.edu/projects/glove/
    from gensim.scripts.glove2word2vec import glove2word2vec

    glove_path = "glove.6B.100d.txt"
    w2v_path = "glove.6B.100d.w2v.txt"
    if not os.path.exists(w2v_path):
        glove2word2vec(glove_path, w2v_path)
    embedding_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    embedding_dim = embedding_model.vector_size

    # Generate embedding matrix
    embedding_matrix = create_embedding_matrix(vocab, embedding_model, embedding_dim)
    np.save("embedding_matrix.npy", embedding_matrix)
    print("Saved embedding_matrix.npy")
