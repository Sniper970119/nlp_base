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

import os
import pickle
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def load_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data_line = f.read()
    data_line = data_line.split('\n')
    text = []
    label = []
    for each in tqdm(data_line):
        if each == '':
            continue
        a, b = each.split('\t')
        text.append(a)
        label.append(int(b))
    return text, label


def handle_data(data, config):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    tokenizer = config.tokenizer
    for each in tqdm(data):
        temp = tokenizer.encode_plus(each, max_length=32, padding='max_length', truncation=True)
        # input_ids.append(np.asarray(temp['input_ids'], dtype=np.int32))
        # token_type_ids.append(np.asarray(temp['token_type_ids'], dtype=np.int32))
        # attention_mask.append(np.asarray(temp['attention_mask'], dtype=np.int32))
        input_ids.append(temp['input_ids'])
        token_type_ids.append(temp['token_type_ids'])
        attention_mask.append(temp['attention_mask'])
    input_ids = np.asarray(input_ids, dtype=np.int32)
    token_type_ids = np.asarray(token_type_ids, dtype=np.int32)
    attention_mask = np.asarray(attention_mask, dtype=np.int32)
    return [input_ids, token_type_ids, attention_mask]


def load_data(config):
    if os.path.exists(config.save_pkl):
        with open(config.save_pkl, 'rb') as f:
            data = pickle.load(f)
        return data['train_text'], data['train_label'], data['dev_text'], data['dev_label'], data['test_text'], data[
            'test_label']
    else:
        train_text, train_label = load_file(config.train_path)
        dev_text, dev_label = load_file(config.dev_path)
        test_text, test_label = load_file(config.test_path)

        train_text = handle_data(train_text, config)
        dev_text = handle_data(dev_text, config)
        test_text = handle_data(test_text, config)

        train_label = tf.keras.utils.to_categorical(train_label, num_classes=config.num_classes)
        dev_label = tf.keras.utils.to_categorical(dev_label, num_classes=config.num_classes)
        test_label = tf.keras.utils.to_categorical(test_label, num_classes=config.num_classes)

        # train_text = np.array(train_text, dtype=np.int32)
        # train_label = np.array(train_label, dtype=np.int32)
        # dev_text = np.array(dev_text, dtype=np.int32)
        dev_label = np.array(dev_label, dtype=np.int32)
        test_text = np.array(test_text, dtype=np.int32)
        test_label = np.array(test_label, dtype=np.int32)

        data = {'train_label': train_label,
                'train_text': train_text,
                'dev_text': dev_text,
                'dev_label': dev_label,
                'test_text': test_text,
                'test_label': test_label,
                }
        with open(config.save_pkl, 'wb') as f:
            pickle.dump(data, f)

        return train_text, train_label, dev_text, dev_label, test_text, test_label


if __name__ == '__main__':
    from summary.models.BERTFC import Config

    c = Config()
    load_data(c)
