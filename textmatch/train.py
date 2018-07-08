#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import importlib
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib.training import HParams

from .preprocessor import Conf
from .utils import get_data, print_hparams


def add_arguments():
    tf.flags.DEFINE_string('classifier', 'BiLSTM', "classifier.the classifier should be registered in model/__init__.py")
    tf.flags.DEFINE_integer('nb_epoch', 50, "number of epoch")
    tf.flags.DEFINE_integer('embed_size', 200, "hidden size of embedding layer")
    tf.flags.DEFINE_integer('batch_size', 320, "batch size for train")
    tf.flags.DEFINE_string('optimizer', 'adam', "the optimizer for train")
    tf.flags.DEFINE_bool('use_pretrained', True, "if use pretrained vector for embedding layer")
    tf.flags.DEFINE_string('class_weight', '1:4', "class weight for 0 and 1 label. ")
    tf.flags.DEFINE_bool('trainable', True,
                         "if the embedding layer is trainable. this param is used only `use_pretrained` is true")

    # data relation
    tf.flags.DEFINE_string('max_len', '20:40',   # 20,word. 40, char. is good
                            "regular sentence to a fixed length. first is word, second is char")

    return tf.flags.FLAGS


def create_hparams(flags):
    return HParams(
        classifier=flags.classifier,
        nb_epoch=flags.nb_epoch,
        embed_size=flags.embed_size,
        batch_size=flags.batch_size,
        optimizer=flags.optimizer,
        use_pretrained=flags.use_pretrained,
        class_weight=dict(enumerate(float(n) for n in flags.class_weight.split(':'))),
        trainable=flags.trainable,

        max_len=map(int, flags.max_len.split(':')),

        time=datetime.now().strftime('%y%m%d%H%M%S'),

        textcnn_filters_char=[(1, 128), (2, 128), (3, 128), (4, 128), (5, 32), (6, 32)],
        textcnn_filters_word=[(1, 128), (2, 128), (3, 128), (4, 128), (5, 32), (6, 32)],
    )


def main(hparams):
    data = get_data(Conf.config.get('DEFAULT', 'raw_train_data'), hparams.max_len, False)

    cls_name = hparams.classifier
    _custom_model_module = 'model'
    _module = importlib.import_module(_custom_model_module)

    try:
        cls = getattr(_module, cls_name)
    except AttributeError:
        raise AttributeError('the model `%s` not found. you should register model in %s.__init__.py' %
                             (cls_name, _custom_model_module))
    print_hparams("Hparams", hparams)
    model = cls(data=data, hparams=hparams)
    model.train()

if __name__ == '__main__':
    FLAGS = add_arguments()
    hparams = create_hparams(FLAGS)
    main(hparams)
