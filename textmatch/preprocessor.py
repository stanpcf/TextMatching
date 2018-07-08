#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, absolute_import, division
import codecs
import glob
import os

import pandas as pd
from six.moves import configparser


class TextPreProcessor(object):

    @staticmethod
    def clean_text(text, word_level):
        """
        :param text: the string of text
        :param word_level: word_level to cut
        :return: text. the text is cleaned and then joined by space
        """
        pass


def get_raw_data(data_pt, pair_tag, label_tag, is_test=False, is_dir=True):
    """
    :param data_pt: train or test data path. or the dir of data.
    :param pair_tag:
    :param label_tag:
    :param is_test:
    :param is_dir: if the data_pt is directory
    :return: dataFrame. text pair cleaned and origin and the label(test for -1 pad)
    """
    from .preprocessor import TextPreProcessor
    if is_dir:
        data_pt = glob.glob(os.path.join(data_pt, '*.csv'))
    else:
        data_pt = [data_pt]
    data = None
    for pt in data_pt:
        tmp = read_csv(pt, pair_tag, label_tag, is_test)
        if data is None:
            data = tmp
        else:
            data = pd.concat([data, tmp])
    data.reset_index(inplace=True)
    data['q1_word'] = data['question1'].apply(lambda x: TextPreProcessor.clean_text(x, True))
    data['q2_word'] = data['question2'].apply(lambda x: TextPreProcessor.clean_text(x, True))
    data['q1_char'] = data['question1'].apply(lambda x: TextPreProcessor.clean_text(x, False))
    data['q2_char'] = data['question2'].apply(lambda x: TextPreProcessor.clean_text(x, False))
    return data


def read_csv(data_pt, pair_tag, label_tag, is_test):
    with codecs.open(data_pt, 'r', encoding='utf-8') as f:
        lnums, q1s, q2s, labels = [], [], [], []
        for i, line in enumerate(f, start=1):
            if not is_test:
                lnum, q1, q2, label = line.strip().split('\t')
            else:
                lnum, q1, q2 = line.strip().split('\t')
            lnums.append(i)
            q1s.append(unicode(q1))
            q2s.append(unicode(q2))
            if not is_test:
                labels.append(int(label))
            else:
                labels.append(-1)
        df = pd.DataFrame({
            pair_tag: lnums,
            'question1': q1s,
            'question2': q2s,
            label_tag: labels
        })
    return df


class Conf(object):
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf/config.conf'))
    pair_tag = 'pair_id'
    label_tag = 'is_duplicate'
