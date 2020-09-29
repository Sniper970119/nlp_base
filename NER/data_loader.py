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
from tqdm import tqdm
from collections import Counter
import pickle
import os
import numpy as np
import itertools

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def load_sentences(path):
    """
    从文档中读取句子
    :param path:
    :return:
    """
    sentences = []
    sentence = []
    for line in tqdm(codecs.open(path, 'r', encoding='utf-8'), desc='数据读取'):
        line = line.strip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) == 2
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def change_bio_to_bioes(sentences):
    """
    将bio编码转换为BIOES编码
    :param sentences:
    :return:
    """
    new_tags = []
    new_sentences = []
    for idx, sentence in tqdm(enumerate(sentences), desc='数据处理'):
        tags = [each[-1] for each in sentence]
        new_tag = ['O']
        for tag in tags:
            # 处理O前的I 改为E
            if tag == 'O' and new_tag[-1].split('-')[0] == 'I':
                new_tag[-1] = 'E-' + str(new_tag[-1].split('-')[1])
            # 处理O前的B 改为S
            if tag == 'O' and new_tag[-1].split('-')[0] == 'B':
                new_tag[-1] = 'S-' + str(new_tag[-1].split('-')[1])
            # 放行O
            if tag == 'O':
                new_tag.append(tag)
            # 放行 B
            if tag.split('-')[0] == 'B':
                new_tag.append(tag)
            # 放行 I
            if tag.split('-')[0] == 'I':
                new_tag.append(tag)
        new_tags.append(new_tag[1:])
        for i in range(len(sentence)):
            # 因为添加了一个O在最前面统一操作，因此+1对齐原句
            sentence[i][-1] = new_tag[i + 1]
        new_sentences.append(sentence)
    return new_sentences


def word_mapping(sentences):
    """
    单词映射，获取所有单词的id映射（这一步很显然是one-hot操作，可以被BERT等词嵌入操作取代）
    :param sentences:
    :return:
    """
    words = []
    for sentence in sentences:
        for each in sentence:
            words.append(each[0])
    words_counter = Counter(words)
    words_dict = {}
    for each in words_counter:
        words_dict[each] = words_counter[each]
    words_dict['<PAD>'] = 10000001
    words_dict['<UNK>'] = 10000000
    sorted_items = sorted(words_dict.items(), key=lambda x: -x[1])
    id_to_word = {i: v[0] for i, v in enumerate(sorted_items)}
    word_to_id = {v[0]: i for i, v in enumerate(sorted_items)}
    return words_dict, id_to_word, word_to_id


def tag_mapping(sentences):
    """
    标签映射，类似于one-hot，这一步不可以取代，类似于输出类别索引
    :param sentences:
    :return:
    """
    tags = []
    for sentence in sentences:
        for each in sentence:
            tags.append(each[1])
    tags_counter = Counter(tags)
    tags_dict = {}
    for each in tags_counter:
        tags_dict[each] = tags_counter[each]
    sorted_item = sorted(tags_dict.items(), key=lambda x: -x[1])
    id_to_tags = {i: v[0] for i, v in enumerate(sorted_item)}
    tags_to_id = {v[0]: i for i, v in enumerate(sorted_item)}
    return tags_dict, id_to_tags, tags_to_id


def prepare_dataset(sentences, word_to_id, tags_to_id):
    """
    准备训练数据，将sentences的word和tag全部转换为idx
    :param sentences:
    :param words_to_id:
    :param tags_to_id:
    :return:
    """
    data = []
    for sentence in sentences:
        word_list = [word[0] for word in sentence]
        # 这里其实有一点问题，这里已经应该是idx了，不应该else UNK,应该是上面的‘10000000’，
        # 但是因为词汇表就是从数据集中获取的，一般不会在这里报错
        word_id_list = [word_to_id[word if word in word_to_id else '<UNK>'] for word in word_list]
        tag_id_list = [tags_to_id[word[-1]] for word in sentence]
        data.append([word_list, word_id_list, tag_id_list])
    return data


def load_data(config):
    """
    读取数据（整合）
    :param config:
    :return:
    """
    # 由于处理需要时间，因此进行序列化存储，这里检查是否有存储过的序列化文件
    if os.path.exists(config.dataset_pkt):
        dataset_pkt = pickle.load(open(config.dataset_pkt, 'rb'))
        train_sentences = dataset_pkt['train']
        dev_sentences = dataset_pkt['dev']
        test_sentences = dataset_pkt['test']
    else:
        # 加载数据集
        train_sentences = load_sentences(config.train_path)
        dev_sentences = load_sentences(config.dev_path)
        test_sentences = load_sentences(config.test_path)
        # 编码转换
        train_sentences = change_bio_to_bioes(train_sentences)
        dev_sentences = change_bio_to_bioes(dev_sentences)
        test_sentences = change_bio_to_bioes(test_sentences)
        dataset_plt = {}
        dataset_plt['train'] = train_sentences
        dataset_plt['dev'] = dev_sentences
        dataset_plt['test'] = test_sentences
        pickle.dump(dataset_plt, open(config.dataset_pkt, 'wb'))

    # 单词映射以及标签映射的存储
    if os.path.exists(config.map_pkt):
        map_pkt = pickle.load(open(config.map_pkt, 'rb'))
        words_dict = map_pkt['words_dict']
        id_to_word = map_pkt['id_to_word']
        word_to_id = map_pkt['word_to_id']
        tags_dict = map_pkt['tags_dict']
        id_to_tags = map_pkt['id_to_tags']
        tags_to_id = map_pkt['tags_to_id']
    else:
        words_dict, id_to_word, word_to_id = word_mapping(train_sentences)
        tags_dict, id_to_tags, tags_to_id = tag_mapping(train_sentences)
        map_pkt = {}
        map_pkt['words_dict'] = words_dict
        map_pkt['id_to_word'] = id_to_word
        map_pkt['word_to_id'] = word_to_id
        map_pkt['tags_dict'] = tags_dict
        map_pkt['id_to_tags'] = id_to_tags
        map_pkt['tags_to_id'] = tags_to_id
        pickle.dump(map_pkt, open(config.map_pkt, 'wb'))

    # 处理后的data文件
    if os.path.exists(config.handled_pkt):
        data_pkt = pickle.load(open(config.handled_pkt, 'rb'))
        train_data = data_pkt['train_data']
        dev_data = data_pkt['dev_data']
        test_data = data_pkt['test_data']
    else:
        train_data = prepare_dataset(train_sentences, word_to_id, tags_to_id)
        test_data = prepare_dataset(test_sentences, word_to_id, tags_to_id)
        dev_data = prepare_dataset(dev_sentences, word_to_id, tags_to_id)
        data_dict = {}
        data_dict['train_data'] = train_data
        data_dict['test_data'] = test_data
        data_dict['dev_data'] = dev_data
        pickle.dump(data_dict, open(config.handled_pkt, 'wb'))

    return train_data, dev_data, test_data, word_to_id, id_to_word, tags_to_id, id_to_tags


def load_word2vec(config, id_to_word):
    """
    读取word2vec词嵌入向量
    :param config:
    :param id_to_word:
    :param word_dim:
    :return:
    """
    if os.path.exists(config.embedding_matrix_file):
        embedding_mat = np.load(config.embedding_matrix_file)
        return embedding_mat
    else:
        pre_trained = {}
        emb_invalid = 0
        for i, line in enumerate(codecs.open(config.emb_file, 'r', encoding='utf-8')):
            line = line.rstrip().split()
            if len(line) == config.embsize + 1:
                pre_trained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:
                emb_invalid = emb_invalid + 1

        if emb_invalid > 0:
            print('waring: %i invalid lines' % emb_invalid)

        num_words = len(id_to_word)
        embedding_mat = np.zeros([num_words, config.embsize])
        for i in range(num_words):
            word = id_to_word[i]
            if word in pre_trained:
                embedding_mat[i] = pre_trained[word]
            else:
                pass
        print('加载了 %i 个字向量' % len(pre_trained))
        np.save(config.embedding_matrix_file, embedding_mat)
        return embedding_mat


def get_X_and_Y_data(dataset, max_len, num_classes):
    """
    将数据拆分为X和Y
    :param dataset:
    :param max_len:
    :param num_classes:
    :return:
    """
    x_data = [data[1] for data in dataset]
    x_data = pad_sequences(x_data, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0)
    y_data = [data[2] for data in dataset]
    y_data = pad_sequences(y_data, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0)
    y_data = to_categorical(y_data, num_classes=num_classes)
    return x_data, y_data


def format_result(chars, tags):
    """
    将网络输出转换为字典格式
    :param chars:
    :param tags:
    :return:
    """
    return {}
    pass


if __name__ == '__main__':
    sentences = load_sentences(r'./data/dev.txt')
    bioes_tag = change_bio_to_bioes(sentences)
    words_dict, id_to_word, word_to_id = word_mapping(sentences)
    tags_dict, id_to_tags, tags_to_id = tag_mapping(sentences)
    data = prepare_dataset(sentences, word_to_id, tags_to_id)
    print()
