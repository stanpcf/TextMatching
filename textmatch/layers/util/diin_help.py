from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.util import nest
# import tensorflow as tf
from keras import backend as K
from .general import flatten, reconstruct, add_wd, exp_mask

# from util.parameters as params
# FIXED_PARAMETERS, config = params.load_parameters()


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    with K.tf.variable_scope(scope or "linear"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        flat_args = [flatten(arg, 1) for arg in args]
        # if input_keep_prob < 1.0:
        assert is_train is not None
        # flat_args = [K.tf.cond(is_train, lambda: K.tf.nn.dropout(arg, input_keep_prob), lambda: arg)
        #              for arg in flat_args]
        flat_args = [K.tf.nn.dropout(arg, input_keep_prob) for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias)
        # flat_out = K.python.ops.rnn_cell._linear(flat_args, output_size, bias)
        out = reconstruct(flat_out, args[0], 1)
        if squeeze:
            out = K.tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])
        # if wd:
        #     add_wd(wd)
    return out

"""
def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        # if keep_prob < 1.0:
        d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        out = tf.cond(is_train, lambda: d, lambda: x)
        return out
        # return x
"""


def softmax(logits, mask=None, scope=None):
    with K.tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = K.tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)
        return out


def softsel(target, logits, mask=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with K.tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = K.tf.reduce_sum(K.tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


def double_linear_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0,
                         is_train=None):
    with tf.variable_scope(scope or "Double_Linear_Logits"):
        first = tf.tanh(linear(args, size, bias, bias_start=bias_start, scope='first',
                               wd=wd, input_keep_prob=input_keep_prob, is_train=is_train))
        second = linear(first, 1, bias, bias_start=bias_start, squeeze=True, scope='second',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            second = exp_mask(second, mask)
        return second


def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with K.tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def sum_logits(args, mask=None, name=None):
    with tf.name_scope(name or "sum_logits"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(tf.reduce_sum(arg, rank - 1) for arg in args)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def get_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None,
               func=None):
    # if func is None:
    #     func = "sum"
    # if func == 'sum':
    #     return sum_logits(args, mask=mask, name=scope)
    # elif func == 'linear':
    #     return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
    #                          input_keep_prob=input_keep_prob,
    #                          is_train=is_train)
    # elif func == 'double':
    #     return double_linear_logits(args, size, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
    #                                 input_keep_prob=input_keep_prob,
    #                                 is_train=is_train)
    # elif func == 'dot':
    #     assert len(args) == 2
    #     arg = args[0] * args[1]
    #     return sum_logits([arg], mask=mask, name=scope)
    # elif func == 'scaled_dot':
    #     assert len(args) == 2
    #     dim = args[0].get_shape().as_list()[-1]
    #     arg = args[0] * args[1]
    #     arg = arg / tf.sqrt(tf.constant(dim, dtype=tf.float32))
    #     return sum_logits([arg], mask=mask, name=scope)
    # elif func == 'mul_linear':
    #     assert len(args) == 2
    #     arg = args[0] * args[1]
    #     return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
    #                          input_keep_prob=input_keep_prob,
    #                          is_train=is_train)
    # elif func == 'proj':
    #     assert len(args) == 2
    #     d = args[1].get_shape()[-1]
    #     proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
    #                   is_train=is_train)
    #     return sum_logits([proj * args[1]], mask=mask)
    # elif func == 'tri_linear':
    assert len(args) == 2
    new_arg = args[0] * args[1]
    return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
                         input_keep_prob=input_keep_prob,
                         is_train=is_train)
    # else:
    #     raise Exception()


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None, output_size=None):
    with tf.variable_scope(scope or "highway_layer"):
        if output_size is not None:
            d = output_size
        else:
            d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob,
                       is_train=is_train)

        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        if d != arg.get_shape()[-1]:
            arg = linear([arg], d, bias, bias_start=bias_start, scope='arg_resize', wd=wd,
                         input_keep_prob=input_keep_prob, is_train=is_train)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None,
                    output_size=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train, output_size=output_size)
            prev = cur
        return cur


def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        # if is_train is not None and keep_prob < 1.0:
        in_ = dropout(in_, keep_prob, is_train)
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob,
                         scope="conv1d_{}".format(height))
            outs.append(out)
        # concat_out = tf.concat(2, outs)
        concat_out = tf.concat(outs, axis=2)
        return concat_out


def conv2d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv2d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        out = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        return out


def cosine_similarity(lfs, rhs):  # [N, d]
    dot = tf.reduce_sum(lfs * rhs, axis=1)
    base = tf.sqrt(tf.reduce_sum(tf.square(lfs), axis=1)) * tf.sqrt(tf.reduce_sum(tf.square(rhs), axis=1))
    return dot / base


def variable_summaries(var, scope):
    """summaries for tensors"""
    with tf.name_scope(scope or 'summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def dense_logits(config, args, out_size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0,
                 is_train=None, func=None):
    with K.tf.variable_scope(scope or "dense_logits"):

        # Tri_linear
        if func == "tri_linear":
            new_arg = args[0] * args[1]
            cat_dim = len(new_arg.get_shape().as_list()) - 1
            cat_args = tf.concat([args[0], args[1], new_arg], axis=cat_dim)
            print("cat args shape")
            print(cat_args.get_shape())

            # if config.dense_logits_with_mask:
            #     out = linear(cat_args, out_size ,True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd, input_keep_prob=config.keep_rate,
            #                             is_train=is_train)
            #     if mask is not None:
            #         mask_exp = tf.cast(tf.tile(tf.expand_dims(mask, 3), [1,1,1, out_size]),tf.float32)
            #         out = mask_exp * out
            #     return out

            out = linear(cat_args, out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)
        elif func == "mul":
            cat_args = args[0] * args[1]

            out = linear([cat_args], out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)

        elif func == "cat_linear":
            cat_dim = len(args[0].get_shape().as_list()) - 1
            cat_args = tf.concat([args[0], args[1]], axis=cat_dim)

            out = linear(cat_args, out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)

        elif func == "diff_mul":
            diff = args[0] - args[1]
            mul = args[0] * args[1]
            cat_dim = len(mul.get_shape().as_list()) - 1
            cat_args = tf.concat([diff, mul], axis=cat_dim)
            out = linear(cat_args, out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)

        elif func == "diff":
            diff = args[0] - args[1]
            out = linear(diff, out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)


        else:
            raise Exception()

        # if config.dense_logits_with_mask:
        #     if mask is not None:
        #         mask_exp = tf.cast(tf.tile(tf.expand_dims(mask, 3), [1,1,1, out_size]),tf.float32)
        #         out = mask_exp * out


        variable_summaries(out, "dense_logits_out_summaries")

        if config.visualize_dense_attention_logits:
            list_of_logits = tf.unstack(out, axis=3)
            for i in range(len(list_of_logits)):
                tf.summary.image("dense_logit_layer_{}".format(i), tf.expand_dims(list_of_logits[i], 3), max_outputs=2)

        return out


def fuse_gate(is_train, lhs, rhs, dr_rate, scope=None):
    with K.tf.variable_scope(scope or "fuse_gate"):
        dim = lhs.get_shape().as_list()[-1]
        lhs_1 = linear(lhs, dim, True, bias_start=0.0, scope="lhs_1", squeeze=False, wd=0.0,
                       input_keep_prob=1-dr_rate, is_train=is_train)
        rhs_1 = linear(rhs, dim, True, bias_start=0.0, scope="rhs_1", squeeze=False, wd=0.0,
                       input_keep_prob=1-dr_rate, is_train=is_train)
        # if config.self_att_fuse_gate_residual_conn and config.self_att_fuse_gate_relu_z:
        #     z = tf.nn.relu(lhs_1 + rhs_1)
        # else:
        z = K.tf.tanh(lhs_1 + rhs_1)
        lhs_2 = linear(lhs, dim, True, bias_start=0.0, scope="lhs_2", squeeze=False, wd=0.0,
                       input_keep_prob=1-dr_rate, is_train=is_train)
        rhs_2 = linear(rhs, dim, True, bias_start=0.0, scope="rhs_2", squeeze=False, wd=0.0,
                       input_keep_prob=1-dr_rate, is_train=is_train)
        f = K.tf.sigmoid(lhs_2 + rhs_2)

        # if config.two_gate_fuse_gate:
        #     lhs_3 = linear(lhs, dim, True, bias_start=0.0, scope="lhs_3", squeeze=False, wd=config.wd,
        #                    input_keep_prob=config.keep_rate, is_train=is_train)
        #     rhs_3 = linear(rhs, dim, True, bias_start=0.0, scope="rhs_3", squeeze=False, wd=config.wd,
        #                    input_keep_prob=config.keep_rate, is_train=is_train)
        #     f2 = tf.sigmoid(lhs_3 + rhs_3)
        #     out = f * lhs + f2 * z
        # else:
        out = f * lhs + (1 - f) * z

        return out

"""
def dense_net(config, denseAttention, is_train):
    with K.tf.variable_scope("dense_net"):
        dim = denseAttention.get_shape().as_list()[-1]
        act = K.tf.nn.relu if config.first_scale_down_layer_relu else None
        fm = K.tf.contrib.layers.convolution2d(denseAttention, int(dim * config.dense_net_first_scale_down_ratio),
                                               config.first_scale_down_kernel, padding="SAME", activation_fn=act)

        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="first_dense_net_block")
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='second_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="second_dense_net_block")
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='third_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="third_dense_net_block")

        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='fourth_transition_layer')

        shape_list = fm.get_shape().as_list()
        print(shape_list)
        out_final = K.tf.reshape(fm, [-1, shape_list[1] * shape_list[2] * shape_list[3]])
        return out_final


def dense_net_block(config, feature_map, growth_rate, layers, kernel_size, is_train, padding="SAME", act=tf.nn.relu,
                    scope=None):
    with K.tf.variable_scope(scope or "dense_net_block"):
        conv2d = K.tf.contrib.layers.convolution2d
        dim = feature_map.get_shape().as_list()[-1]

        list_of_features = [feature_map]
        features = feature_map
        for i in range(layers):
            ft = conv2d(features, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=act)
            list_of_features.append(ft)
            features = K.tf.concat(list_of_features, axis=3)

        print("dense net block out shape")
        print(features.get_shape().as_list())
        return features


def dense_net_transition_layer(config, feature_map, transition_rate, scope=None):
    with K.tf.variable_scope(scope or "transition_layer"):
        out_dim = int(feature_map.get_shape().as_list()[-1] * transition_rate)
        feature_map = K.tf.contrib.layers.convolution2d(feature_map, out_dim, 1, padding="SAME", activation_fn=None)

        feature_map = K.tf.nn.max_pool(feature_map, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

        print("Transition Layer out shape")
        print(feature_map.get_shape().as_list())
        return feature_map
"""