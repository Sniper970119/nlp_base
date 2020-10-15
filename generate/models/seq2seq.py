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

        self.num_samples = 1000000
        self.num_epochs = 50
        self.batch_size = 64
        self.embedding_dim = 300
        self.hidden_size = 256
        self.steps_per_epoch = 50
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.max_encoder_seq_length = 0
        self.max_decoder_seq_length = 0
        self.input_length = 0


class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config

        # self.encoder = GRUEncoder(self.config.num_encoder_tokens, self.config.embedding_dim, self.config.hidden_size,
        #                           config.batch_size)
        # self.decoder = GRUDecoder(self.config.num_decoder_tokens, self.config.embedding_dim, self.config.hidden_size,
        #                           config.batch_size)

    def my_model(self):
        encoder = GRUEncoder(self.config.num_encoder_tokens, self.config.embedding_dim, self.config.hidden_size,
                             self.config.batch_size)
        decoder = GRUDecoder(self.config.num_decoder_tokens, self.config.embedding_dim, self.config.hidden_size,
                             self.config.batch_size)

        return encoder, decoder

    # def call(self, input, enc_hidden, tokenizer):
    #     enc_outout, enc_hidden = self.encoder(input, enc_hidden)
    #     dec_hidden = enc_hidden
    #     dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * self.config.batch_size, 1)
    #
    #     for t in range(1, targ.shape[1]):
    #         # 将编码器输出 （enc_output） 传送至解码器
    #         predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
    #
    #         loss += loss_function(targ[:, t], predictions)
    #         x = targ[:, t]
    #         dec_input = tf.expand_dims(x, 1)
    #     pass
