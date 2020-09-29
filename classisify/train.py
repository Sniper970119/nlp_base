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
import os
import time
from importlib import import_module
from summary.utils import build_dataset, get_time_dif, build_net_data
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='model config')
parser.add_argument('--model', default='text_LSTM_2', type=str)
parser.add_argument('--word', default=False, type=bool)
args = parser.parse_args()

if __name__ == '__main__':
    embedding = 'embedding_SougouNews.npz'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(embedding)
    start_time = time.time()
    print('loading data...')
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    config.num_of_vocab = len(vocab)
    time_dif = get_time_dif(start_time)
    print('time usage:', time_dif)

    train_x, train_y = build_net_data(train_data, config)
    dev_x, dev_y = build_net_data(dev_data, config)
    test_x, test_y = build_net_data(test_data, config)

    model = x.MyModel(config)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    logdir = os.path.join("tensorboard")
    callbacks = [
        tf.keras.callbacks.TensorBoard(logdir),
        tf.keras.callbacks.ModelCheckpoint(filepath=config.save_path_ckpt, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
    ]

    history = model.fit(x=train_x, y=train_y, validation_data=(dev_x, dev_y), batch_size=512, epochs=2,
                        callbacks=callbacks)
    print(history.history)
    preds = model.evaluate(test_x, test_y, batch_size=64)
    print('test acc:', preds)