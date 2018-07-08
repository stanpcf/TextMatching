# coding: utf-8
from __future__ import absolute_import, print_function, division
from tensorflow.python.ops.nn import softmax
from tensorflow.python.util import nest
from .util.general import flatten, reconstruct
from keras.layers import Layer, Activation
from keras import backend as K
from keras import initializers, regularizers, constraints


class ESIMAttention(Layer):
    """see: https://arxiv.org/abs/1609.06038 at `3.2 Local Inference Modeling` """
    def __init__(self, **kwargs):
        super(ESIMAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('`%s` layer should be called on a list of 2 inputs.' % (self.__class__.__name__,))
        shape1, shape2 = input_shape[0], input_shape[1]
        assert len(shape1) == 3 and len(shape2) == 3 and shape1[-1] == shape2[-1], \
            'tensor input shape not correct. tensor shape: %s' % str(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x1, x2 = inputs[0], inputs[1]
        e = K.batch_dot(x1, x2, axes=[2, 2])
        e1 = softmax(e, 2)
        e2 = softmax(e, 1)
        xe1 = K.batch_dot(e1, x2, axes=[2, 1])
        xe2 = K.batch_dot(e2, x1, axes=[2, 1])
        return [xe1, xe2]


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

    def get_config(self):
        config = {
            'step_dim': self.step_dim,
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
            'b_constraint': constraints.serialize(self.b_constraint),
            'bias': self.bias
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# build layer for diin
class SelfAttention(Layer):
    def __init__(self, max_len, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        sent_shape, _ = input_shape
        shape1 = sent_shape[:2] + sent_shape[1:]
        shape2 = sent_shape[:2] + sent_shape[1:]
        mask_shape = list(sent_shape[:2]) + [sent_shape[1]]
        return [shape1, shape2, tuple(mask_shape)]

    def call(self, inputs, **kwargs):
        sent, sent_mask = inputs[0], inputs[1]
        sent_max_len = self.max_len
        sent_mask = K.tf.expand_dims(sent_mask, 2)
        p_aug_1 = K.tf.tile(K.tf.expand_dims(sent, 2), [1, 1, sent_max_len, 1])
        p_aug_2 = K.tf.tile(K.tf.expand_dims(sent, 1), [1, sent_max_len, 1, 1])
        p_mask_aug_1 = K.tf.reduce_any(K.tf.cast(K.tf.tile(K.tf.expand_dims(sent_mask, 2), [1, 1, sent_max_len, 1]), K.tf.bool), axis=3)
        p_mask_aug_2 = K.tf.reduce_any(K.tf.cast(K.tf.tile(K.tf.expand_dims(sent_mask, 1), [1, sent_max_len, 1, 1]), K.tf.bool), axis=3)
        self_mask = p_mask_aug_1 & p_mask_aug_2  # [none, 42, 42]
        return [p_aug_1, p_aug_2, self_mask]
        # h_logits = self.get_logits([p_aug_1, p_aug_2], True, mask=self_mask)
        # self_att = softsel(p_aug_2, h_logits)
        # return fuse_gate(True, sent, self_att, 0.1, scope='self_att_fuse_gate')

    # def get_logits(self, args, bias, mask=None, input_keep_prob=1.0):
    #     new_arg = args[0] * args[1]
    #     logits = self.my_linear([args[0], args[1], new_arg], 1, bias, scope='first', input_keep_prob=input_keep_prob)
    #     if mask is not None:
    #         logits = exp_mask(logits, mask)
    #     return logits

    def get_config(self):
        config = {
            'max_len': self.max_len
        }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FuseGate(Layer):
    def __init__(self, dim, **kwargs):
        super(FuseGate, self).__init__(**kwargs)
        self.dim = dim
        self.lhs1 = MyLinear(dim)
        self.rhs1 = MyLinear(dim)
        self.lhs2 = MyLinear(dim)
        self.rhs2 = MyLinear(dim)

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        self_att_shape = input_shape[1]
        return tuple(list(self_att_shape)[:2] + [self.dim])

    def call(self, inputs, **kwargs):
        sent, self_att = inputs[0], inputs[1]
        z = Activation('tanh')(self.lhs1(sent) + self.rhs1(self_att))
        f = Activation('sigmoid')(self.lhs2(sent) + self.rhs2(self_att))
        return f * sent + (1-f) * z


class MyLinear(Layer):
    def __init__(self, units, dropout=0.0, squeeze=False, **kwargs):
        super(MyLinear, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.squeeze = squeeze

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            x = input_shape[0]
        else:
            x = input_shape
        output_shape = list(x[:-1])
        if not self.squeeze:
            output_shape.append(self.units)
        return tuple(output_shape)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            x = input_shape
        else:
            x = [input_shape]
        ls = 0
        print(input_shape)
        for sp in x: ls += sp[-1]
        self._linear_weights = self.add_weight(name='kernel', shape=(ls, self.units),
                                               initializer=initializers.get('glorot_uniform'))
        self._linear_bias = self.add_weight(name='bias', shape=(self.units,), initializer=initializers.get('zeros'))
        self.built = True

    def call(self, inputs, **kwargs):
        if inputs is None or (nest.is_sequence(inputs) and not inputs):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(inputs):
            inputs = [inputs]
        flat_args = [flatten(arg, 1) for arg in inputs]
        flat_args = [K.tf.nn.dropout(arg, 1-self.dropout) for arg in flat_args]
        flat_out = self._linear(flat_args, True)
        out = reconstruct(flat_out, inputs[0], 1)
        if self.squeeze:
            out = K.tf.squeeze(out, [len(inputs[0].get_shape().as_list())-1])
        return out

    def _linear(self, args, bias):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

        Args:
          args: a 2D Tensor or a list of 2D, batch x n, Tensors.
          # output_size: int, second dimension of W[i].
          bias: boolean, whether to add a bias term or not.

        Returns:
          A 2D Tensor with shape [batch x output_size] equal to
          sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

        Raises:
          ValueError: if some of the arguments has unspecified or wrong shape.
        """
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        if len(args) == 1:
            res = K.tf.matmul(args[0], self._linear_weights)
        else:
            res = K.tf.matmul(K.concatenate(args, 1), self._linear_weights)
        if not bias:
            return res
        return K.bias_add(res, self._linear_bias)


class InteractionLayer(Layer):
    def __init__(self, **kwargs):
        super(InteractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        p, h = inputs[0], inputs[1]
        PL = p.get_shape().as_list()[1]
        HL = h.get_shape().as_list()[1]
        p_aug = K.tf.tile(K.tf.expand_dims(p, 2), [1, 1, HL, 1])
        h_aug = K.tf.tile(K.tf.expand_dims(h, 1), [1, PL, 1, 1])  # [N, PL, HL, 2d]
        h_logits = p_aug * h_aug
        return h_logits

    def compute_output_shape(self, input_shape):
        p_shape, h_shape = input_shape[0], input_shape[1]
        return p_shape[:2] + h_shape[1:]
