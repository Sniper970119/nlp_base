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
import time

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from translation.models.transformer import Transformer
from translation.models.transformer import Config
from translation.utils import *
from translation.models.model_utils import *

if __name__ == '__main__':
    train_input_text, val_input_text, train_target_text, val_target_text, en_tokenizer, zh_tokenizer = \
        load_dataset('./data/en-zh.csv')

    dataset = tf.data.Dataset.from_tensor_slices((val_input_text, val_target_text)).shuffle(len(train_input_text))

    config = Config()
    config.input_vocab_size = len(en_tokenizer.index_word)
    config.target_vocab_size = len(zh_tokenizer.index_word)
    dataset = dataset.batch(1, drop_remainder=True)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


    def loss_function(y_true, y_pred):
        # 识别掩码位置
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        _loss = loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=_loss.dtype)
        _loss *= mask
        return tf.reduce_mean(_loss)


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

    transformer = Transformer(config.num_layers, config.d_model, config.num_heads, config.dff, config.input_vocab_size,
                              config.target_vocab_size, config.max_seq_len, config.dropout_rate)


    def create_mask(inputs, targets):
        encode_padding_mask = create_padding_mark(inputs)
        decode_padding_mask = create_padding_mark(inputs)

        look_ahead_mask = create_look_ahead_mark(tf.shape(targets)[1])
        decode_target_padding_mask = create_padding_mark(targets)

        combine_mask = tf.maximum(decode_target_padding_mask, look_ahead_mask)

        return encode_padding_mask, combine_mask, decode_padding_mask


    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=1000):
            super(CustomSchedule, self).__init__()

            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


    learning_rate = CustomSchedule(config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    # optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, config.save_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('last checkpoint restore')


    def train_step(inputs, targets):
        target_input = targets[:, :-1]
        target_real = targets[:, 1:]

        encode_padding_mask, combined_mask, decode_padding_mask = create_mask(inputs, target_input)

        # with tf.GradientTape() as tape:
        prediction, _ = transformer(inputs, target_input,
                                        False,
                                        encode_padding_mask,
                                        combined_mask,
                                        decode_padding_mask)
        loss = loss_function(target_real, prediction)

        out = []
        for each in prediction[0]:
            idx = np.argmax(each)
            out.append(idx)
            pass
        output_text = zh_tokenizer.sequences_to_texts(np.array([out]))
        print(output_text)
        print(en_tokenizer.sequences_to_texts(np.array([inputs[0]])))
        print()

        train_loss(loss)
        train_acc(target_real, prediction)


    for batch, (inputs, targets) in enumerate((dataset)):
        train_step(inputs, targets)

        if batch % 200 == 0:
            print('batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                batch, train_loss.result(), train_acc.result()
            ))