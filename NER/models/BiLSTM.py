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


class Config():
    def __init__(self):
        self.model_name = 'BiLSTM'
        self.train_path = r'./data/train.txt'
        self.dev_path = r'./data/dev.txt'
        self.test_path = r'./data/test.txt'
        self.dataset_pkt = r'./handled_data/dataset.pkt'
        self.map_pkt = r'./handled_data/map.pkt'
        self.handled_pkt = r'./handled_data/handled.pkt'
        self.save_path = r'./output/' + self.model_name + '.ckpt'
        self.emb_file = r'./data/wiki_100.utf8'
        self.embedding_matrix_file = r'./data/word_embedding_matrix.npy'
        self.embsize = 100

        self.tags_num = 13  # O*1  LOC*4   PER*4  ORG*4 (B I E S)

        self.dropout = 0.5
        self.num_epochs = 10
        self.max_len = 200
        self.learning_rate = 1e-3
        self.hidden_size = 128


class MyModel(tf.keras.Model):
    def __init__(self, config, embedding_pretrained):
        super(MyModel, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(input_dim=embedding_pretrained.shape[0],
                                                   output_dim=embedding_pretrained.shape[1],
                                                   input_length=self.config.max_len,
                                                   weights=[embedding_pretrained],
                                                   trainable=True
                                                   )
        self.BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.config.hidden_size,
                                                                         return_sequences=True,
                                                                         activation='relu'
                                                                         ))
        self.dropout = tf.keras.layers.Dropout(self.config.dropout)
        self.out = tf.keras.layers.Dense(self.config.tags_num, activation='softmax')

    def call(self, x):
        x = self.embedding(x)
        x = self.BiLSTM(x)
        x = self.dropout(x)
        x = self.out(x)
        return x
