# coding: utf-8
# ESIM https://arxiv.org/abs/1609.06038
# official paper code impl: https://github.com/lukecq1231/nli

from .base_model import *


class ESIM(TextModel):
    # todo: impl the tree-LSTM for ESIM

    def get_model(self, trainable=None):
        _embed = get_embedding_layer(self.data.char_index, max_len=self.hparams.max_len[1],
                                     embedding_dim=self.hparams.embed_size, word_level=False,
                                     use_pretrained=self.hparams.use_pretrained, trainable=self.hparams.trainable)
        lstm_embed = lambda tensor: Bidirectional(LSTM(128, return_sequences=True))(SpatialDropout1D(0.5)(tensor))

        input1 = Input(shape=(self.hparams.max_len[1],), name='q1_char')
        input2 = Input(shape=(self.hparams.max_len[1],), name='q2_char')

        seq1, seq2 = _embed(input1), _embed(input2)
        seq1, seq2 = lstm_embed(seq1), lstm_embed(seq2)

        xe1, xe2 = ESIMAttention()([seq1, seq2])

        conc_embed = lambda _t1, _t2: Bidirectional(LSTM(128, return_sequences=True))(
            concatenate([_t1, _t2,
                         Lambda(lambda x: K.abs(x[0] - x[1]))([_t1, _t2]),
                         Lambda(lambda x: x[0] * x[1])([_t1, _t2])]
                        ))
        m1, m2 = conc_embed(seq1, xe1), conc_embed(seq2, xe2)
        x = concatenate([GlobalMaxPool1D()(m1), GlobalAvgPool1D()(m1), GlobalMaxPool1D()(m2), GlobalAvgPool1D()(m2)])
        x = Dense(128, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        # x = Dense(2, activation=softmax)(x)
        pred = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[input1, input2], outputs=pred)
        return model


def subtract(input_1, input_2):
    minus_input_2 = Lambda(lambda x: -x)(input_2)
    return add([input_1, minus_input_2])


def aggregate(input_1, input_2, num_dense=300, dropout_rate=0.5):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])
    x = concatenate([feat1, feat2])
    x = BatchNormalization()(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x


def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, dim=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, dim=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


class ESIM1(TextModel):
    def get_model(self, trainable=None):
        q1 = Input(shape=(self.hparams.max_len[1],), name='q1_char')
        q2 = Input(shape=(self.hparams.max_len[1],), name='q2_char')

        # Embedding
        embedding = get_embedding_layer(self.data.char_index, max_len=self.hparams.max_len[1],
                                        embedding_dim=self.hparams.embed_size, word_level=False,
                                        use_pretrained=self.hparams.use_pretrained, trainable=self.hparams.trainable)
        q1_embed = BatchNormalization(axis=2)(embedding(q1))
        q2_embed = BatchNormalization(axis=2)(embedding(q2))

        # Encoding
        encode = Bidirectional(LSTM(128, return_sequences=True))
        q1_encoded = encode(q1_embed)
        q2_encoded = encode(q2_embed)

        # Alignment
        q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)

        # Compare
        q1_combined = concatenate(
            [q1_encoded, q2_aligned, subtract(q1_encoded, q2_aligned), multiply([q1_encoded, q2_aligned])])
        q2_combined = concatenate(
            [q2_encoded, q1_aligned, subtract(q2_encoded, q1_aligned), multiply([q2_encoded, q1_aligned])])
        compare = Bidirectional(LSTM(128, return_sequences=True))
        q1_compare = compare(q1_combined)
        q2_compare = compare(q2_combined)

        # Aggregate
        x = aggregate(q1_compare, q2_compare)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=[q1, q2], outputs=x)
