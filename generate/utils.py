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

import codecs
import tensorflow as tf


def add_BOS_EOS(sentence: str):
    """
    向句子中 添加EOS、BOS标记
    :param sentence:
    :return:
    """
    words = sentence.strip()
    w = '<BOS> ' + words + ' <EOS>'
    return w


def load_file(path: str):
    """
    从文件中读取数据
    :param path:
    :param nub_example:
    :return:
    """
    lines = codecs.open(path, encoding='utf-8').read().strip().split('\n')
    return lines


def split_data(lines: list, nub_example: int):
    """
    将从文件中读取的数据处理切割
    :param lines:
    :param nub_example:
    :return:
    """
    sentences = [[add_BOS_EOS(sentence) for sentence in each.split('#')] for each in lines[:nub_example]]
    return zip(*sentences)


def tokenize(sentences: list):
    """
    创建词汇表
    :param sentences:
    :return:
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def get_idx(tokenizer, text, padding_size=50):
    """
    根据词汇表获取idx（token）
    :param tokenizer:
    :param text:
    :return:
    """
    tensor = tokenizer.texts_to_sequences(text)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=padding_size, padding='post')
    return tensor


def load_data(path, nub_example):
    """
    方法整合，读取数据
    :param path:
    :param nub_example:
    :return:
    """
    lines = load_file(path=path)
    input_text, target_text = split_data(lines, nub_example)
    tokenizer = tokenize(input_text + target_text)
    input_text = get_idx(tokenizer, input_text)
    target_text = get_idx(tokenizer, target_text)
    return input_text, target_text, tokenizer


if __name__ == '__main__':
    load_data('./data/train.txt', 100)
