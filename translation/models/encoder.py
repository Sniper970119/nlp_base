# -*- coding:utf-8 -*-

"""
      ┏┛ ┻━━━━━┛ ┻┓
      ┃　　　　　　 ┃
      ┃　　　━　　　┃
      ┃　┳┛　  ┗┳　┃
      ┃　　　　　　 ┃
      ┃　　　┻　　　┃
      ┃　　　　　　 ┃
      ┗━┓　　　┏━━━┛
        ┃　　　┃   神兽保佑
        ┃　　　┃   代码无BUG！
        ┃　　　┗━━━━━━━━━┓
        ┃　　　　　　　    ┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛
"""

import tensorflow as tf

from translation.models.encoderLayer import EncoderLayer
from translation.models.decoderLayer import DecoderLayer
from translation.models.model_utils import *


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf, input_vocab_size, max_seq_len, dropout_rate=0.1):
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.encoder_layer = [EncoderLayer(d_model, n_heads, ddf, dropout_rate) for _ in range(n_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mark):
        seq_len = inputs.shape[1]
        word_emb = self.embedding(inputs)
        word_emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        emb = word_emb + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encoder_layer[i](x, training, mark)
        return x


if __name__ == '__main__':
    sample_encoder = Encoder(2, 512, 8, 1024, 5000, 200)
    sample_encoder_output = sample_encoder(tf.random.uniform((64, 120)),
                                           False, None)
    print(sample_encoder_output.shape)
