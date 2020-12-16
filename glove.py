import os

import numpy as np
from tqdm.auto import tqdm


# GLOVE_DIR = path for glove.6B.100d.txt
def glove_dictionary(GLOVE_DIR, dim=200):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, f'glove.6B.{dim}d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def embedding_matrix_creator(embedding_dim, word2idx, GLOVE_DIR='data/glove.6B/'):
    embeddings_index = glove_dictionary(GLOVE_DIR=GLOVE_DIR, dim=embedding_dim)
    embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    for word, i in tqdm(word2idx.items()):
        embedding_vector = embeddings_index.get(word.lower())
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
