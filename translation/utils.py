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
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split


def add_BOS_and_EOS(sentence):
    """
    增加BOS EOS标记
    :param sentence:
    :return:
    """
    sentence = sentence.lower().strip()
    sentence = '<BOS>' + sentence + '<EOS>'
    return sentence


def load_file(file_name, example_nub):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    if example_nub is None:
        example_nub = len(lines)
    datas = [[add_BOS_and_EOS(s) for s in l.split('\t')] for l in lines[:example_nub]]
    return zip(*datas)


def get_tokenizer(text):
    """
    获取词汇表
    :param text:
    :return:
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(text)
    return tokenizer


def get_text_idx(text, en_tokenizer=None, zh_tokenizer=None, flag='en', padding_size=50):
    """
    将文本 映射成idx
    :param text:
    :return:
    """
    assert flag in ['en', 'zh']
    assert flag == 'zh' or en_tokenizer
    assert flag == 'en' or zh_tokenizer

    if flag == 'en':
        tensor = en_tokenizer.texts_to_sequences(text)
    else:
        tensor = zh_tokenizer.texts_to_sequences(text)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=padding_size, padding='post')
    return tensor


def load_dataset(path, config, num_example=None):
    """
    创建数据集
    :param path:
    :param config:
    :param num_example:
    :return:
    """
    input_text, target_text = load_file(path, num_example)

    en_tokenizer = get_tokenizer(input_text)
    zh_tokenizer = get_tokenizer(target_text)

    input_text = get_text_idx(input_text, en_tokenizer=en_tokenizer, zh_tokenizer=zh_tokenizer, flag='en',
                              padding_size=50)
    target_text = get_text_idx(target_text, en_tokenizer=en_tokenizer, zh_tokenizer=zh_tokenizer, flag='zh',
                               padding_size=50)

    train_input_text, val_input_text, train_target_text, val_target_text = train_test_split(input_text, target_text,
                                                                                            test_size=0.2)
    return train_input_text, val_input_text, train_target_text, val_target_text, en_tokenizer, zh_tokenizer


if __name__ == '__main__':
    load_dataset('./data/en-zh.csv', 100)
