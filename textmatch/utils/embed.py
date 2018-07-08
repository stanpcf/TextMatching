# coding: utf8

import os

import numpy as np
from keras.layers import Embedding
from ..preprocessor import Conf


def get_embedding_layer(word_index, max_len=100, embedding_dim=100, use_pretrained=True, trainable=True,
                        word_level=False):
    """
    :param word_index: word_index from tokenizer
    :param max_len:
    :param embedding_dim:
    :param use_pretrained: if use pretrained vector to init embedding layer. default true
    :param trainable:
    :param word_level:
    :return: keras embedding layer
    """
    num_words = len(word_index) + 1

    if not use_pretrained:
        return Embedding(num_words, embedding_dim, input_length=max_len)
    else:
        initial_weights = _get_pretrain_weights(num_words, word_index, embedding_dim, word_level)
        return Embedding(num_words, embedding_dim, weights=[initial_weights], input_length=max_len, trainable=trainable)


def _get_pretrain_weights(num_words, word_index, dim, word_level):
    """
    :param word_index: token.word_index
    :param dim: 维度
    :return: embedding 权重
    """
    embeddings_index = {}
    W2V_DIR = Conf.config.get("DEFAULT", "word2vec_pt")
    filename = "w2v_{dim}_50_{_wind}_{wl}.txt".format(dim=dim, _wind=3, wl=int(word_level))
    filename = os.path.join(W2V_DIR, filename)
    with open(filename) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((num_words, dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
