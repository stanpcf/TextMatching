# coding: utf-8
from .base_model import *


class TextRNN(TextModel):

    def get_model(self, trainable=None):
        _embed = get_embedding_layer(self.data.word_index, max_len=self.hparams.max_len, embedding_dim=self.hparams.embed_size,
                                     use_pretrained=self.hparams.use_pretrained, trainable=trainable)
        _bi_lstm = Bidirectional(GRU(128))   # 128. 0.45
        seq_embed = lambda tensor: _bi_lstm(_embed(tensor))  # 两个句子公用一个lstm和embedding权重

        input1 = Input(shape=(self.hparams.max_len,), name='question1')
        input2 = Input(shape=(self.hparams.max_len,), name='question2')
        seq1, seq2 = seq_embed(input1), seq_embed(input2)

        diff = Lambda(lambda x: K.abs(x[0] - x[1]))([seq1, seq2])
        mul = Lambda(lambda x: x[0] * x[1])([seq1, seq2])

        x = concatenate([diff, mul])

        # x = Dropout(0.5)(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        pred = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[input1, input2], outputs=pred)
        return model
