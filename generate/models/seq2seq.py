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

from generate.models.GRUDecoder import GRUDecoder
from generate.models.GRUEncoder import GRUEncoder


class Config():
    def __init__(self):
        self.model_name = 'seq2seq_attention'
        self.train_path = './data/train.txt'
        self.test_path = './data/test.txt'
        self.save_path = './output/' + self.model_name + '.ckpt'

        self.nub_sameples = 10000
        self.num_epochs = 20
        self.batch_size = 64
        self.embedding_dim = 300
        self.hidden_size = 256
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.max_encoder_seq_length = 0
        self.max_decoder_seq_length = 0
        self.input_length = 0


class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config

    def my_model(self):
        encoder = GRUEncoder(self.config.num_encoder_tokens, self.config.embedding_dim, self.config.hidden_size,
                             self.config.batch_size)
        decoder = GRUDecoder(self.config.num_decoder_tokens, self.config.embedding_dim, self.config.hidden_size,
                             self.config.batch_size)

        return encoder, decoder
