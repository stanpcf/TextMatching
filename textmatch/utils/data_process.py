# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
from sklearn.utils import Bunch
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer  # 这几个不存在tf和keras行为不一致问题。不用选择

from ..preprocessor import Conf, get_raw_data
from ..utils import LogUtil


def _get_tokenizer():
    train_data = get_raw_data(Conf.config.get("DEFAULT", "raw_train_data"),
                              pair_tag=Conf.pair_tag, label_tag=Conf.label_tag, is_test=False)
    tokenizer_word = Tokenizer()
    tokenizer_word.fit_on_texts(np.hstack([train_data['q1_word'].values, train_data['q2_word'].values]))
    tokenizer_char = Tokenizer()
    tokenizer_char.fit_on_texts(np.hstack([train_data['q1_char'].values, train_data['q2_char'].values]))
    return tokenizer_word, tokenizer_char


def _to_seq(data, tokenizer, tag, max_len, is_reverse):
    sent_seq = tokenizer.texts_to_sequences(data[tag].values)
    if is_reverse:
        sent_seq = [list(reversed(li)) for li in sent_seq]
    return pad_sequences(sent_seq, maxlen=max_len, padding='post', truncating='post'), np.array(map(len, sent_seq))


def get_data(data_pt, max_len, is_test):
    """
    :param data_pt: str: train or test data_path.
    :param max_len: int: pad length. should be 2-tuple
    :param is_test:

    sep_reverse: 是否构造一份reverse文本的数据集。这样做的好处是可以防止尾部信息损失。同时使用分开的数据,在构造attention的时候效果应该比不分开的bi-lstm好
    :return: dict:
        word_level:
            q1_word, q2_word: question-pair
            q1_word_r, q2_word_r: question-pair reversed
            q1_word_len, q2_word_len: question len
            word_index: word to index
        char_level: same as word_level
        label
    """

    data = get_raw_data(data_pt, pair_tag=Conf.pair_tag, label_tag=Conf.label_tag, is_test=is_test,
                        is_dir=not is_test)
    tokenizer_word, tokenizer_char = _get_tokenizer()
    label = data[Conf.label_tag].values if not is_test else None

    result = Bunch(label=label)
    result['word_index'] = tokenizer_word.word_index
    result['q1_word'], result['q1_word_len'] = _to_seq(data, tokenizer_word, 'q1_word', max_len[0], is_reverse=False)
    result['q2_word'], result['q2_word_len'] = _to_seq(data, tokenizer_word, 'q2_word', max_len[0], is_reverse=False)
    result['q1_word_r'], _ = _to_seq(data, tokenizer_word, 'q1_word', max_len[0], is_reverse=True)
    result['q2_word_r'], _ = _to_seq(data, tokenizer_word, 'q2_word', max_len[0], is_reverse=True)

    result['char_index'] = tokenizer_char.word_index
    result['q1_char'], result['q1_char_len'] = _to_seq(data, tokenizer_char, 'q1_char', max_len[1], is_reverse=False)
    result['q2_char'], result['q2_char_len'] = _to_seq(data, tokenizer_char, 'q2_char', max_len[1], is_reverse=False)
    result['q1_char_r'], _ = _to_seq(data, tokenizer_char, 'q1_char', max_len[1], is_reverse=True)
    result['q2_char_r'], _ = _to_seq(data, tokenizer_char, 'q2_char', max_len[1], is_reverse=True)

    LogUtil.log("INFO", "load data done!")
    return result
