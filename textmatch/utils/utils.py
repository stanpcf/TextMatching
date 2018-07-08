# coding: utf-8
from __future__ import print_function

import tensorflow as tf
import codecs
import json
import time
import sys


def print_hparams(header, hparams):
    """Print hparams, can skip keys based on pattern."""
    print(header)
    values = hparams.values()
    for key in sorted(values.keys()):
        print("\t{0}={1}".format(key, values[key]))


def load_hparams(hparams_file):
    """Load hparams from an existing model directory."""
    if tf.gfile.Exists(hparams_file):
        print("# Loading hparams from %s" % hparams_file)
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
            except ValueError:
                print("can't load hparams file")
                return None
        if hasattr(hparams, 'class_weight'):    # 处理tensorflow在json_dumps后会将分类权重的key(int)转化为str的这个bug
            cw = hparams.class_weight
            hparams.class_weight = dict([(int(i), cw[i]) for i in cw.keys()])
        return hparams
    else:
        return None


def save_hparams(out_path, hparams):
    """Save hparams."""
    print("saving hparams to %s" % out_path)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(out_path, "wb")) as f:
        f.write(hparams.to_json())


def save_history(out_path, history):
    """Save keras history."""
    with codecs.getwriter("utf-8")(tf.gfile.GFile(out_path, "wb")) as f:
        f.write(str(history.history))


class TimeUtil(object):
    """
    tool for log
    """
    @staticmethod
    def time_now():
        """
        Get the current time, e.g. `2016-12-27 17:14:01`
        :return: string represented current time
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    @staticmethod
    def time_now_YmdH():
        return time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))


class LogUtil(object):
    """
    tool for log
    """

    @staticmethod
    def log(typ, msg):
        """
        print message for log
        :param typ: type of log
        :param msg: str: message of log.
        :return: None
        """
        print("{time}\t{typ}\t{msg}".format(time=TimeUtil.time_now(), typ=typ, msg=msg))
        sys.stdout.flush()
