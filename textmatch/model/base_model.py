# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from random import randint

from sklearn.model_selection import train_test_split
from sklearn.utils import safe_indexing
from tensorflow.python.ops.nn import softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras.layers import *

from ..preprocessor import Conf
from ..utils import pdb
from ..layers import *
from ..utils import f1_score as NN_F1, precision as NN_P, recall as NN_R
from ..utils import save_hparams, save_history, get_embedding_layer


CUSTOM_OBJECTS = {'f1_score': NN_F1, 'precision': NN_P, 'recall': NN_R, 'softmax': softmax}
CUSTOM_OBJECTS.update(CUSTOM_LAYERS)


class TextModel(object):
    """abstract base model for all text model."""
    __metaclass__ = ABCMeta

    def __init__(self, data, hparams):
        """
        :param data: data_process.get_data返回的对象
        :param hparams:
        """
        self.data = data
        self.hparams = hparams
        self.name = self.__class__.__name__
        if self.hparams.is_kfold:
            self.bst_model_path_list = []

    @abstractmethod
    def get_model(self, trainable=None):
        """定义一个keras net, compile it and return the model"""
        raise NotImplementedError

    def _get_bst_model_path(self):
        """return a name which is used for save trained weights"""
        return "{pre}_{time}".format(pre=self.__class__.__name__, time=self.hparams.time)

    def get_bst_model_path(self, is_retrain=False):
        dirname = os.path.join(Conf.config.get('DEFAULT', 'model_save_pt'), self.__class__.__name__)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        sub_fix = "_retrain" if is_retrain else ""
        return os.path.join(dirname, self._get_bst_model_path() + sub_fix)

    def train(self):
        data = deepcopy(self.data)
        bst_model_path = self.get_bst_model_path(is_retrain=False)
        early_stopping = EarlyStopping(monitor='val_f1_score', patience=5, mode='max')
        model_checkpoint = ModelCheckpoint(bst_model_path+'.h5', save_best_only=True, save_weights_only=False)
        hist_file = bst_model_path + ".history"

        model = self.get_model(trainable=True)
        model.summary()
        if self.__class__.__name__ == 'MatchPyramid':
            data['pw_index'] = DynamicMaxPooling.dynamic_pooling_index(data.q1_word_len, data.q2_word_len,
                                                                       self.hparams.max_len[0], self.hparams.max_len[0])
            data['pc_index'] = DynamicMaxPooling.dynamic_pooling_index(data.q1_char_len, data.q2_char_len,
                                                                       self.hparams.max_len[1], self.hparams.max_len[1])

        _tr_idx, _te_idx = train_test_split(range(data.label.shape[0]), random_state=2017, test_size=0.1)
        train_y, valid_y = safe_indexing(data.label, _tr_idx), safe_indexing(data.label, _te_idx)
        for k in ['label', 'word_index', 'char_index']: data.pop(k)
        train_x, valid_x = {}, {}
        for k, v in data.iteritems():
            train_x[k], valid_x[k] = safe_indexing(v, _tr_idx), safe_indexing(v, _te_idx)
        model.compile(loss='binary_crossentropy', optimizer=self.hparams.optimizer, metrics=['acc', NN_F1, NN_P, NN_R])
        hist = model.fit_generator(self.batch_data_generater(train_x, train_y, self.hparams.batch_size),
                                   epochs=self.hparams.nb_epoch, validation_data=(valid_x, valid_y),
                                   callbacks=[model_checkpoint, early_stopping],
                                   class_weight=self.hparams.class_weight,
                                   steps_per_epoch=train_y.shape[0] // self.hparams.batch_size)
        save_history(hist_file, hist)
        save_hparams(bst_model_path+'.hparams', self.hparams)

    @classmethod
    def batch_data_generater(cls, train_x, train_y, batch_size):
        _len = train_y.shape[0]
        loopcount = _len // batch_size
        keys = train_x.keys()
        while 1:
            i = randint(0, loopcount)
            yield dict([(key, train_x[key][i * batch_size: (i + 1) * batch_size]) for key in keys]), \
                  train_y[i * batch_size:(i + 1) * batch_size]

    @classmethod
    def model_predict(cls, model_path, data, hparams, layer_name=''):
        """
        :param model_path: 保存的训练好的模型权重的文件
        :param data: 测试数据
        :param hparams:
        :param layer_name: 某一层的名字
        :return:
        """
        model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        if hparams.classifier == 'MatchPyramid':
            data['pw_index'] = DynamicMaxPooling.dynamic_pooling_index(data.q1_word_len, data.q2_word_len,
                                                                       hparams.max_len[0], hparams.max_len[0])
            data['pc_index'] = DynamicMaxPooling.dynamic_pooling_index(data.q1_char_len, data.q2_char_len,
                                                                       hparams.max_len[1], hparams.max_len[1])

        if not layer_name or len(layer_name) == '':
            prob = model.predict(data)[:, 0]
            return (prob >= 0.5).astype(int)
        else:
            x = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(data)
            pdb.set_trace()
            return x
