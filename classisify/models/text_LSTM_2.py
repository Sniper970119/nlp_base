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
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Flatten, Bidirectional
import numpy as np


class Config():
    def __init__(self, embedding):
        self.model_name = 'TextLSTM_2'
        self.train_path = r'./data/train.txt'
        self.dev_path = r'./data/dev.txt'
        self.test_path = r'./data/test.txt'
        self.class_list = [x.strip() for x in open(r'./data/class.txt', 'r').readlines()]
        self.vocab_path = r'./data/vocab.data'
        self.save_path_ckpt = r'./output/' + self.model_name + '.ckpt'
        self.log_path = r'./log/' + self.model_name + '.txt'
        self.embedding_pretrain = np.load(r'./data/' + embedding)['embeddings'].astype(np.float32)

        self.dropout = 0.5
        self.num_classes = len(self.class_list)
        self.num_of_vocab = 0
        self.num_epochs = 10
        self.batch_size = 128
        self.max_len = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrain.shape[1]
        self.hidden_size = 128
        self.num_layers = 2


class MyModel(tf.keras.Model):
    def __init__(self, config, trainable=True):
        super(MyModel, self).__init__()
        self.config = config
        self.embedding = Embedding(input_dim=self.config.embedding_pretrain.shape[0],
                                   output_dim=self.config.embedding_pretrain.shape[1],
                                   input_length=self.config.max_len,
                                   weights=[self.config.embedding_pretrain],
                                   trainable=trainable
                                   )
        self._LSTM = Bidirectional(LSTM(units=self.config.hidden_size,
                                        return_sequences=True,
                                        activation='relu'
                                        ))
        self.dropout = Dropout(self.config.dropout)

        self.flatten = Flatten()

        self._output = Dense(units=self.config.num_classes, activation='softmax')

    def build(self, input_shape):
        super(MyModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self._LSTM(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self._output(x)
        return x
