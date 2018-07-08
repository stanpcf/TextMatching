# coding: utf-8

from .base_model import *
from ..layers.util.general import exp_mask
from ..layers.util.diin_help import softsel


class DIIN(TextModel):
    def get_model(self, trainable=None):
        embedding = get_embedding_layer(self.data.char_index, max_len=self.hparams.max_len[1],
                                        embedding_dim=self.hparams.embed_size,
                                        use_pretrained=self.hparams.use_pretrained, trainable=trainable,
                                        word_level=False)
        query = Input(shape=(self.hparams.max_len[1],), name='q1_char')
        doc = Input(shape=(self.hparams.max_len[1],), name='q2_char')
        query_len = Input(name='q1_char_len', shape=(1,))  # 虽然输入数据是(batch,). 但是这儿还是这个。因为keras会将(batch,)数据扩展维度为(batch,1)
        doc_len = Input(name='q2_char_len', shape=(1,))

        q_mask = SequenceMask(self.hparams.max_len[1])(query_len)
        d_mask = SequenceMask(self.hparams.max_len[1])(doc_len)

        seq1 = embedding(query)
        seq2 = embedding(doc)

        lstm_hid = 128

        context_layer = ContextLayer(lstm_hid, return_sequences=True,
                                     input_shape=(self.hparams.max_len[1], K.int_shape(seq1)[-1],))

        seq1 = context_layer(seq1)
        seq2 = context_layer(seq2)

        self_att = SelfAttention(max_len=self.hparams.max_len[1])

        seq1_p_aug_1, seq1_p_aug_2, seq1_self_mask = self_att([seq1, q_mask])
        seq1_new_aug = multiply([seq1_p_aug_1, seq1_p_aug_2])
        seq1_logits = MyLinear(1, squeeze=True)([seq1_p_aug_1, seq1_p_aug_2, seq1_new_aug])
        seq1_logits = Lambda(lambda x: exp_mask(x[0], x[1]),
                             output_shape=lambda input_shape: input_shape[1])([seq1_logits, seq1_self_mask])
        seq1_self_att = Lambda(lambda x: softsel(x[0], x[1]),
                               output_shape=lambda input_shape: tuple(list([input_shape[0][0]])+list(input_shape[0][-2:])))([seq1_p_aug_2, seq1_logits])
        seq1 = FuseGate(lstm_hid*2)([seq1, seq1_self_att])

        seq2_p_aug_1, seq2_p_aug_2, seq2_self_mask = self_att([seq2, d_mask])
        seq2_new_aug = multiply([seq2_p_aug_1, seq2_p_aug_2])
        seq2_logits = MyLinear(1, squeeze=True)([seq2_p_aug_1, seq2_p_aug_2, seq2_new_aug])
        seq2_logits = Lambda(lambda x: exp_mask(x[0], x[1]),
                             output_shape=lambda input_shape: input_shape[1])([seq2_logits, seq2_self_mask])
        seq2_self_att = Lambda(lambda x: softsel(x[0], x[1]),
                               output_shape=lambda input_shape: tuple(list([input_shape[0][0]])+list(input_shape[0][-2:])))([seq2_p_aug_2, seq2_logits])
        seq2 = FuseGate(lstm_hid*2)([seq2, seq2_self_att])

        matrix = InteractionLayer()([seq1, seq2])
        matrix = Dropout(0.1)(matrix)  # [none, len, len, dim]
        matrix = Conv2D(100, [2, 2], strides=[2, 2], activation='relu')(matrix)  # 接下来的相当于图片的特征抓取。
        matrix = Conv2D(64, [2, 2], strides=[2, 2], activation='relu')(matrix)  # 接下来的相当于图片的特征抓取。
        matrix = Conv2D(32, [2, 2], strides=[2, 2], activation='relu')(matrix)  # 接下来的相当于图片的特征抓取。

        flat_matrix = Flatten()(matrix)

        x = Dense(64, activation='relu')(flat_matrix)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[query, doc, query_len, doc_len], outputs=x)
        return model
