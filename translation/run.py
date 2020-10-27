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

    dataset = tf.data.Dataset.from_tensor_slices((train_input_text, train_target_text)).shuffle(len(train_input_text))

    config = Config()
    config.input_vocab_size = len(en_tokenizer.index_word)
    config.target_vocab_size = len(zh_tokenizer.index_word)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


    def loss_function(y_true, y_pred):
        # 识别掩码位置
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        _loss = loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=_loss.dtype)
        _loss += mask
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


    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, config.save_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('last checkpoit restore')


    def train_step(inputs, targets):
        target_input = targets[:, :-1]
        target_real = targets[:, 1:]

        encode_padding_mask, combined_mask, decode_padding_mask = create_mask(inputs, target_input)

        with tf.GradientTape() as tape:
            prediction, _ = transformer(inputs, target_input,
                                        True,
                                        encode_padding_mask,
                                        combined_mask,
                                        decode_padding_mask)
            loss = loss_function(target_real, prediction)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_acc(target_real, prediction)


    for epoch in range(config.epochs):
        start = time.time()

        train_acc.reset_states()
        train_loss.reset_states()

        for batch, (inputs, targets) in enumerate((dataset)):
            train_step(inputs, targets)

        #     if batch % 200 == 0:
        #         print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
        #             epoch + 1, batch, train_loss.result(), train_acc.result()
        #         ))
        # if (epoch + 1) % 2 == 0:
        #     ckpt_save_path = ckpt_manager.save()
        #     print('epoch {}, save model at {}'.format(
        #         epoch + 1, ckpt_save_path
        #     ))
        #
        # print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
        #     epoch + 1, train_loss.result(), train_acc.result()
        # ))
        #
        # print('time in 1 epoch:{} secs\n'.format(time.time() - start))
