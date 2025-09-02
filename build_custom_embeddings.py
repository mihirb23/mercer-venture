import os
import json
from gensim.models import Word2Vec
import numpy as np

# Gather sentences (tokenized) from your own cleaned corpus
sentences = []
for fname in os.listdir("processed_documents"):
    if fname.endswith("_cleaned.txt"):
        with open(os.path.join("processed_documents", fname), encoding="utf-8") as f:
            text = f.read()
            tokens = text.lower().split()  # or better: use spaCy for tokenization
            sentences.append(tokens)

# Train your own Word2Vec model
embedding_dim = 100  # or 200, as you wish
w2v_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
w2v_model.save("custom_word2vec.model")
print("Trained Word2Vec and saved custom_word2vec.model")

# Build vocab: assign index to every word
vocab = {"<PAD>": 0, "<UNK>": 1}
for word in w2v_model.wv.index_to_key:
    if word not in vocab:
        vocab[word] = len(vocab)

with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

# Build embedding matrix
embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
for word, idx in vocab.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
    elif word == "<UNK>":
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    # <PAD> stays zeros

np.save("embedding_matrix.npy", embedding_matrix)
print("Saved vocab.json and embedding_matrix.npy")
