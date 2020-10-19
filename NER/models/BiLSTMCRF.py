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

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'BilSTMCRF'
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
        self.num_epochs = 20
        self.max_len = 200
        self.learning_rate = 1e-3
        self.hidden_size = 128

class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.transition_params = None

        self.embedding = tf.keras.layers.Embedding(config.n_vocab, config.embsize)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.hidden_size, return_sequences=True))
        self.dense = tf.keras.layers.Dense(config.tags_num)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(config.tags_num, config.tags_num)),
                                             trainable=False)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, text,labels=None,training=None):
        x = tf.math.not_equal(text, 0)
        x = tf.cast(x, dtype=tf.int32)
        text_lens = tf.math.reduce_sum(x, axis=-1)
        # -1 change 0
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        inputs = self.biLSTM(inputs)
        logits = self.dense(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits, label_sequences, text_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens
