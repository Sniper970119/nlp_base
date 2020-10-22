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
    datas = [[add_BOS_and_EOS(s) for s in l.split('\t')] for l in lines[:example_nub]]
    return zip(*datas)


if __name__ == '__main__':
    load_file('./data/en-zh.csv', 100)
