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
import argparse
from importlib import import_module
import json
import numpy as np

from NER.data_loader import load_data, load_word2vec, format_result, get_X_and_Y_data

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='Chinese NER')
parser.add_argument('--model_name', default='BiLSTM', type=str, help='model name')

if __name__ == '__main__':
    args = parser.parse_args()

    model_name = args.model_name

    struct_file = import_module('models.' + model_name)

    config = struct_file.Config()

    print('Loading data....')

    train_data, dev_data, test_data, word_to_id, id_to_word, tag_to_id, id_to_tag = load_data(config)
    embedding_pretrained = load_word2vec(config, id_to_word)

    print('Load data success')

    # train
    model = struct_file.MyModel(config, embedding_pretrained)

    model.build(input_shape=(None, config.max_len))
    model.load_weights(config.save_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    test_X, test_y = get_X_and_Y_data(test_data, config.max_len, len(tag_to_id))
    preds = model.evaluate(test_X, test_y, batch_size=64)
    print('test acc:', preds)

    while True:
        text = input("input:")
        dataset = tf.keras.preprocessing.sequence.pad_sequences([[word_to_id.get(char, 0) for char in text]],
                                                                padding='post')
        print(dataset)
        logits = model.predict(dataset)

        for index in logits:
            viterbi_path = np.argmax(index, axis=-1)
        print(viterbi_path)
        print([id_to_tag[id] for id in viterbi_path])

        entities_result = format_result(list(text), [id_to_tag[id] for id in viterbi_path])
        print(json.dumps(entities_result, indent=4, ensure_ascii=False))

