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
        ┃CREATE BY SNIPER┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛

"""

import tensorflow as tf
import transformers


class Config():
    def __init__(self):
        self.model_name = 'BERT+FC'

        self.train_path = r'/data/train.txt'

        self.test_path = r'/data/test.txt'

        self.val_path = r'/data/dev.txt'

        self.save_pkl = '/data/dataset.pkl'

        self.tokenizer = transformers.BertTokenizer.from_pretrained('')

        self.class_list = []

        with open('/data/class.txt', 'r', encoding='utf-8') as f:
            self.class_list = [x.strip() for x in f.readlines()]

        self.save_path = 'output' + self.model_name + '.ckpt'

        self.num_classes = len(self.class_list)

        self.num_epochs = 5

        self.batch_size = 128

        self.max_len = 32

        self.learn_rete = 1e-5


class BERTFC(tf.keras.Model):
    def __init__(self):
        super(BERTFC, self).__init__()
