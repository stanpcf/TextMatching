# coding: utf-8

from .base_model import *


class TextRCNN(TextModel):
    def get_model(self, trainable=None):
        _embed = get_embedding_layer(self.data.char_index, max_len=self.hparams.max_len[1],
                                     embedding_dim=self.hparams.embed_size, word_level=False,
                                     use_pretrained=self.hparams.use_pretrained, trainable=trainable)
        rnn_emb = lambda tensor: Bidirectional(LSTM(128, return_sequences=True))(tensor)

        _convs = [Conv1D(nf, kernel_size=fs, activation='relu', padding='same') for fs, nf in
                  self.hparams.textcnn_filters_char]
        cnn_pool = lambda tensor: concatenate(
            [concatenate([GlobalAvgPool1D()(conv(tensor)), GlobalMaxPool1D()(conv(tensor))])
             for conv in _convs], axis=1)

        input1 = Input(shape=(self.hparams.max_len[1],), name='q1_char')
        input2 = Input(shape=(self.hparams.max_len[1],), name='q2_char')
        seq1, seq2 = _embed(input1), _embed(input2)
        rnn1, rnn2 = rnn_emb(seq1), rnn_emb(seq2)
        seq1, seq2 = concatenate([seq1, rnn1]), concatenate([seq2, rnn2])

        seq1, seq2 = cnn_pool(seq1), cnn_pool(seq2)

        diff = Lambda(lambda x: K.abs(x[0] - x[1]), name='diff')([seq1, seq2])
        mul = Lambda(lambda x: x[0] * x[1], name='mul')([seq1, seq2])
        max = Lambda(lambda x: K.maximum(x[0], x[1]), name='max')([seq1, seq2])
        x = concatenate([diff, mul, max])

        x = Dropout(0.5)(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dropout(0.2)(x)
        pred = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[input1, input2], outputs=pred)
        return model
