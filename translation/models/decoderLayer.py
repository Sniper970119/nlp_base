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

from translation.models.mutiHeadAttention import MutiHeadAttention
from translation.models.model_utils import feed_forward_network


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dff, dropout_rate=0.1):
        """

        :param d_model:
        :param n_heads:
        :param dff:
        :param dropout_rate:
        """
        super(DecoderLayer, self).__init__()

        self.muti_head_attn1 = MutiHeadAttention(d_model, n_heads)
        self.muti_head_attn2 = MutiHeadAttention(d_model, n_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        attn1, attn1_weight = self.muti_head_attn1(inputs, inputs, inputs, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layerNorm1(inputs + attn1)

        attn2, attn2_weight = self.muti_head_attn2(inputs, encoder_out, encoder_out, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layerNorm2(out1 + attn2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layerNorm3(out2 + ffn_out)

        return out3, attn1_weight, attn2_weight


if __name__ == '__main__':
    from translation.models.encoderLayer import EncoderLayer

    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)
    print(sample_decoder_layer_output.shape)
