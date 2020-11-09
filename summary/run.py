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
import time
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from summary.models.BERTFC import BERTFC, Config
from summary.utils import *

if __name__ == '__main__':
    config = Config()
    train_text, train_label, dev_text, dev_label, test_text, test_label = load_data(config)
    bert = BERTFC(config)

    # class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    #     def __init__(self, d_model, warmup_steps=1000):
    #         super(CustomSchedule, self).__init__()
    #
    #         self.d_model = tf.cast(d_model, tf.float32)
    #         self.warmup_steps = warmup_steps
    #
    #     def __call__(self, step):
    #         arg1 = tf.math.rsqrt(step)
    #         arg2 = step * (self.warmup_steps ** -1.5)
    #
    #         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    #
    #
    # learning_rate = CustomSchedule(10)
    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
    #                                      beta_2=0.98, epsilon=1e-9)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-12)

    bert.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    bert.fit(x=train_text, y=train_label, validation_data=(dev_text, dev_label),batch_size=32, epochs=config.num_epochs)

    ckpt = tf.train.Checkpoint(model=bert, optimizer=optimizer)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, config.save_path, max_to_keep=3)
    #
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print('restore')
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    #
    # dataset = tf.data.Dataset.from_tensor_slices((train_text, train_label))
    #
    # loss_fun = tf.keras.losses.CategoricalCrossentropy()
    #
    #
    # def train_step(inputs, targets):
    #     with tf.GradientTape() as tape:
    #         out = bert(inputs)
    #         loss = loss_fun(out, targets)
    #
    #     gradients = tape.gradient(loss, bert.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, bert.trainable_variables))
    #
    #     train_loss(loss)
    #     train_acc(out,targets)
    #
    # for epoch in range(config.num_epochs):
    #     start = time.time()
    #
    #     train_acc.reset_states()
    #     train_loss.reset_states()
    #
    #     for batch, (inputs, targets) in enumerate((dataset)):
    #         train_step(inputs, targets)
    #
    #         if batch % 200 == 0:
    #             print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
    #                 epoch + 1, batch, train_loss.result(), train_acc.result()
    #             ))
    #     if (epoch + 1) % 1 == 0:
    #         ckpt_save_path = ckpt_manager.save()
    #         print('epoch {}, save model at {}'.format(
    #             epoch + 1, ckpt_save_path
    #         ))
    #
    #     print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
    #         epoch + 1, train_loss.result(), train_acc.result()
    #     ))
    #
    #     print('time in 1 epoch:{} secs\n'.format(time.time() - start))
    #
