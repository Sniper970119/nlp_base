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

from translation.utils import *
from translation.models.model_utils import *
from translation.models.transformer import Config, Transformer

if __name__ == '__main__':

    train_input_text, val_input_text, train_target_text, val_target_text, en_tokenizer, zh_tokenizer = \
        load_dataset('./data/en-zh.csv')

    config = Config()
    transformer = Transformer(config.num_layers, config.d_model, config.num_heads, config.dff, config.input_vocab_size,
                              config.target_vocab_size, config.max_seq_len, config.dropout_rate)


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
