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
import numpy as np


def get_angle(pos, i, d_model):
    """
    获得角度（用于获得位置嵌入）
    :param pos:
    :param i:
    :param d_model:
    :return:
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2) / np.float32(d_model)))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    位置嵌入
    :param position:
    :param d_model:
    :return:
    """
    # np.newaxis 的作用是插入新维度
    angle_reds = get_angle(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)

    sin = np.sin(angle_reds[:, 0::2])
    cos = np.cos(angle_reds[:, 1::2])
    pos_encoding = np.concatenate([sin, cos], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# pos_encoding = positional_encoding(50, 512)
# print(pos_encoding.shape)
#
# import matplotlib.pyplot as plt
# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show() # 在这里左右边分别为原来2i 和 2i+1的特征


def create_padding_mark(sequence):
    """
    遮蔽padding
    :param sequence:
    :return:
    """
    sequence = tf.cast(tf.math.equal(sequence, 0), tf.float32)
    return sequence[:, np.newaxis, np.newaxis, :]


def create_look_ahead_mark(size):
    """
    由于不能向后获取信息，当前位置后的信息应该被mark掉
    :param size:
    :return:
    """
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark


def feed_forward_network(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
