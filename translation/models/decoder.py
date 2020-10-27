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


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.decode_layers = [DecoderLayer(d_model, n_heads, ddf, dropout_rate) for _ in range(n_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_out, training, look_ahead_mark, padding_mark):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        h = self.embedding(inputs)
        h *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        h += self.pos_embedding[:, :seq_len, :]
        h = self.dropout(h, training=training)

        for i in range(self.n_layers):
            h, attn_w1, attn_w2 = self.decode_layers[i](h, encoder_out, training, look_ahead_mark, padding_mark)
            attention_weights['decoder_layer{}_attn_w1'.format(i + 1)] = attn_w1
            attention_weights['decoder_layer{}_attn_w2'.format(i + 1)] = attn_w2

        return h, attention_weights


if __name__ == '__main__':
    from translation.models.encoder import Encoder

    sample_encoder = Encoder(2, 512, 8, 1024, 5000, 200)
    sample_encoder_output = sample_encoder(tf.random.uniform((64, 120)),
                                           False, None)
    sample_decoder = Decoder(2, 512, 8, 1024, 5000, 200)
    sample_decoder_output, attn = sample_decoder(tf.random.uniform((64, 100)),
                                                 sample_encoder_output, False,
                                                 None, None)
    print(sample_decoder_output.shape, attn['decoder_layer1_attn_w1'].shape)
