# coding: utf-8

from .base_model import *


class MatchPyramid(TextModel):

    def get_model(self, trainable=None):
        _embed = get_embedding_layer(self.data.char_index, max_len=self.hparams.max_len[1], embedding_dim=self.hparams.embed_size,
                                     use_pretrained=self.hparams.use_pretrained, trainable=trainable)
        input1 = Input(shape=(self.hparams.max_len[1],), name='q1_char')
        input2 = Input(shape=(self.hparams.max_len[1],), name='q2_char')
        pc_index = Input(name='pc_index', shape=[self.hparams.max_len[1], self.hparams.max_len[1], 3], dtype='int32')

        seq1, seq2 = _embed(input1), _embed(input2)

        # cross = MyDot(axes=[2, 2], normalize=False, name='dot')([seq1, seq2])   # 这个地方的数据可能是200维乘积之后导致数据太大。这个地方第解决方案是将每个[max_len, max_len]进行bn
        cross = MyDot(axes=[2, 2], normalize=True, name='dot')([seq1, seq2])   # 这个地方的数据可能是200维乘积之后导致数据太大。这个地方第解决方案是将每个[max_len, max_len]进行bn
        cross_reshape = Reshape((self.hparams.max_len[1], self.hparams.max_len[1], 1))(cross)

        conv2d = Conv2D(64, (5, 5), padding='same', activation='relu')
        dpool = DynamicMaxPooling(5, 5)

        conv1 = conv2d(cross_reshape)
        pool1 = dpool([conv1, pc_index])
        pool1_flat = Flatten()(pool1)

        x = Dropout(0.2)(pool1_flat)
        x = Dense(64, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        pred = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input1, input2, pc_index], outputs=pred)
        return model


class MatchPyramid1(TextModel):
    """ char level"""
    def get_model(self, trainable=None):
        _embed = get_embedding_layer(self.data.char_index, max_len=self.hparams.max_len[1], embedding_dim=self.hparams.embed_size,
                                     use_pretrained=self.hparams.use_pretrained, trainable=trainable, word_level=False)
        _convs = [Conv1D(nf, kernel_size=fs, activation='relu', padding='same') for fs, nf in self.hparams.textcnn_filters_char]

        input1 = Input(shape=(self.hparams.max_len[1],), name='q1_char')
        input2 = Input(shape=(self.hparams.max_len[1],), name='q2_char')
        seq1, seq2 = _embed(input1), _embed(input2)
        cross = concatenate([Reshape((self.hparams.max_len[1], self.hparams.max_len[1], 1))
                             (MyDot(axes=[2, 2], normalize=True)([conv(seq2), conv(seq2)])) for conv in _convs], axis=3)

        conv2d = Conv2D(128, (5, 5), padding='same', activation='relu')(cross)
        x = MaxPool2D()(conv2d)
        x = Flatten()(x)
        # diff = Lambda(lambda x: K.abs(x[0] - x[1]), name='diff')([seq1, seq2])
        # mul = Lambda(lambda x: x[0] * x[1], name='mul')([seq1, seq2])
        # max = Lambda(lambda x: K.maximum(x[0], x[1]), name='max')([seq1, seq2])
        # x = concatenate([diff, mul, max])
        x = Dropout(0.5)(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dropout(0.2)(x)
        pred = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[input1, input2], outputs=pred)
        return model
