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
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import timedelta

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """

    :param file_path:
    :param tokenizer:
    :param max_size:
    :param min_freq:
    :return:
    """
    vocab_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1
        vocab_dict = sorted([_ for _ in vocab_dict.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_dict)}
        vocab_dict.update({UNK: len(vocab_dict)})
    return vocab_dict


def build_dataset(config, use_word):
    """

    :param config:
    :param use_word:
    :return:
    """
    if use_word:
        tokenizer = lambda x: x.split(' ')
    else:
        tokenizer = lambda x: [y for y in x]
        if os.path.exists(config.vocab_path):
            vocab = pickle.load(open(config.vocab_path, 'rb'))
        else:
            vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            pickle.dump(vocab, open(config.vocab_path, 'wb'))
        print('Vocab size:', len(vocab))

    def load_dataset(path):
        """

        :param path:
        :return:
        """
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                word_line = []
                token = tokenizer(content)
                seq_len = len(token)
                for word in token:
                    word_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((word_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return vocab, train, dev, test


def build_net_data(dataset, config):
    """

    :param dataset:
    :param config:
    :return:
    """
    data = [x[0] for x in dataset]
    data_x = pad_sequences(data, maxlen=config.max_len)
    label_y = [x[1] for x in dataset]
    label_y = tf.keras.utils.to_categorical(label_y, num_classes=config.num_classes)
    return data_x, label_y


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

