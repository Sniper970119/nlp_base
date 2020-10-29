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
from translation.models.encoder import Encoder
from translation.models.decoder import Decoder


class Config():
    def __init__(self):
        self.model_name = 'transformer'
        self.epochs = 10
        self.learning_rate = 1e9
        self.batch_size = 32

        self.num_layers = 4
        self.d_model = 128
        self.dff = 512
        self.num_heads = 8

        self.input_vocab_size = 0
        self.target_vocab_size = 0

        self.max_seq_len = 50
        self.dropout_rate = 0.1

        self.save_path = 'output/'


class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, diff, input_vocab_size, target_vocab_size, max_seq_len,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_layers, d_model, n_heads, diff, input_vocab_size, max_seq_len, dropout_rate)
        self.decoder = Decoder(n_layers, d_model, n_heads, diff, target_vocab_size, max_seq_len, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training, encode_padding_mask, look_ahead_mask, decode_padding_mask):
        encode_out = self.encoder(inputs, training, encode_padding_mask)
        # print(encode_out.shape)
        decode_out, attn_weight = self.decoder(targets, encode_out, training, look_ahead_mask, decode_padding_mask)
        # print(decode_out.shape)
        final_out = self.final_layer(decode_out)
        return final_out, attn_weight


if __name__ == '__main__':
    sample_transformer = Transformer(
        n_layers=2, d_model=512, n_heads=8, diff=1024,
        input_vocab_size=8500, target_vocab_size=8000, max_seq_len=120
    )
    temp_input = tf.random.uniform((64, 62))
    temp_target = tf.random.uniform((64, 26))
    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   encode_padding_mask=None,
                                   look_ahead_mask=None,
                                   decode_padding_mask=None,
                                   )
    print(fn_out.shape)
