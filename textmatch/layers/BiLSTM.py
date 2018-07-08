# coding: utf8
from __future__ import print_function
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer
from keras.layers import LSTM, Bidirectional, CuDNNLSTM


class BiLSTM(Layer):
    """
    input shape: (None, time_steps, hid_dim)
    output shape: (None, time_steps, 2*hid_dim), (None, 2*hid_dim)
    """
    def __init__(self, units, dropout=0., **kwargs):
        super(BiLSTM, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout

    def build(self, input_shape):
        # Used purely for shape validation.
        super(BiLSTM, self).build(input_shape)

    def call(self, input, **kwargs):
        # bilstm = Bidirectional(LSTM(self.))
        forward_lstm = CuDNNLSTM(self.units,
                            return_sequences=True, return_state=True)
        backward_lstm = CuDNNLSTM(self.units,
                             return_sequences=True, return_state=True, go_backwards=True)
        fw_outputs, fw_output, fw_state = forward_lstm(input)
        bw_outputs, bw_output, b_state = backward_lstm(input)
        bw_outputs = K.reverse(bw_outputs, 1)
        # bw_output = bw_state[0]
        outputs = K.concatenate([fw_outputs, bw_outputs])
        last_output = K.concatenate([fw_output, bw_output])
        return [outputs, last_output]

    def compute_output_shape(self, input_shape):
        outputs_shape = (input_shape[0], input_shape[1], 2 * self.units)
        output_shape = (input_shape[0], 2 * self.units)
        return [outputs_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'units': self.units,
            'dropout': self.dropout,
        }
        base_config = super(BiLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

