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
import argparse
from importlib import import_module
from sklearn.model_selection import train_test_split

from generate.utils import load_data, add_BOS_EOS
from generate.models.seq2seq import MyModel

parser = argparse.ArgumentParser()

parser.add_argument('--model', default="seq2seq", type=str, help='choose a model: Seq2Seq')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model
    struct_file = import_module('models.' + model_name)
    config = struct_file.Config()

    input_tensor, target_tensor, tokenizer = load_data(config.train_path, config.num_samples)

    train_x, val_x, train_y, val_y = train_test_split(input_tensor, target_tensor, test_size=0.2)

    model = struct_file.MyModel(config)

    encoder, decoder = model.my_model()
