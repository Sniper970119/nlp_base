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


class MutiHeadAttention(tf.keras.layers.Layer):
    """
    多头注意力
    """

    def __init__(self, d_model, num_heads):
        """

        :param d_model: muti attn 输出的维度
        :param num_heads: 多头个数
        """
        super().__init__()

        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model

        self.deep = self.d_model // self.num_heads

        self.WQ = tf.keras.layers.Dense(self.d_model)
        self.WK = tf.keras.layers.Dense(self.d_model)
        self.WV = tf.keras.layers.Dense(self.d_model)

        self.dense = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        """
        分头，
        :param x:
        :param batch_size:
        :return:
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.deep))
        # perm 是新的维度索引，假如x维度是[1,2,3,4]，perm=[0,2,1,3]转置后x维度将变成[1,3,2,4]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.WQ(q)
        k = self.WQ(k)
        v = self.WQ(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention_output, attention_weight = dot_attention(q, k, v, mask)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        output = self.dense(attention_output)
        return output, attention_weight


def dot_attention(q, k, v, mask):
    """
    点乘
    :param q:
    :param k:
    :param v:
    :param mask:
    :return:
    """
    qk = tf.matmul(q, k, transpose_b=True)

    # 计算根号下dk对qk进行缩放，类似于 emm 归一化
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    return output, attention_weights


if __name__ == '__main__':
    temp_mha = MutiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))
    output, att = temp_mha(y, y, y, mask=None)
    print(output.shape, att.shape)
