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
import argparse
from importlib import import_module
from NER.data_loader import load_data, load_word2vec, get_X_and_Y_data

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='Chinese NER')
parser.add_argument('--model_name', default='BiLSTM', type=str, help='model name')

if __name__ == '__main__':
    args = parser.parse_args()

    embedding = 'embedding_SougouNews.npz'
    model_name = args.model_name

    struct_file = import_module('models.' + model_name)

    config = struct_file.Config()

    print('Loading data....')

    train_data, dev_data, test_data, _, id_to_word, tag_to_id, _ = load_data(config)
    embedding_pretrained = load_word2vec(config, id_to_word)

    print('Load data success')

    train_X, train_y = get_X_and_Y_data(train_data, config.max_len, len(tag_to_id))
    dev_X, dev_y = get_X_and_Y_data(dev_data, config.max_len, len(tag_to_id))
    test_X, test_y = get_X_and_Y_data(test_data, config.max_len, len(tag_to_id))

    model = struct_file.MyModel(config, embedding_pretrained)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=config.save_path, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
    ]

    history = model.fit(
        x=train_X,
        y=train_y,
        validation_data=(dev_X, dev_y),
        batch_size=512,
        epochs=config.num_epochs,
        callbacks=callbacks
    )

    preds = model.evaluate(test_X, test_y, batch_size=64)
    print('test acc:', preds)


