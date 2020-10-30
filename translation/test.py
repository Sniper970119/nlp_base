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
    test_text = "Where do you work now ?"
    target_text = "你 现在 在 哪里 工作 ？"
    data = tuple([add_BOS_and_EOS(test_text)])
    target_data = tuple([add_BOS_and_EOS(target_text)])

    train_input_text, val_input_text, train_target_text, val_target_text, en_tokenizer, zh_tokenizer = \
        load_dataset('./data/en-zh.csv')

    input_text = get_text_idx(data, en_tokenizer=en_tokenizer, zh_tokenizer=zh_tokenizer, flag='en',
                              padding_size=50)

    target_text = get_text_idx(target_data, en_tokenizer=en_tokenizer, zh_tokenizer=zh_tokenizer, flag='zh',
                               padding_size=50)

    target_text = target_text[:, :-1]

    input_text = tf.cast(input_text, dtype=tf.int32)
    target_text = tf.cast(target_text, dtype=tf.int32)

    zh_eos_flag = zh_tokenizer.word_index['eos']

    config = Config()
    config.input_vocab_size = len(en_tokenizer.index_word)
    config.target_vocab_size = len(zh_tokenizer.index_word)

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

    encode_padding_mask, combined_mask, decode_padding_mask = create_mask(input_text, target_text)

    prediction, _ = transformer(input_text, target_text,
                                False,
                                encode_padding_mask,
                                combined_mask,
                                decode_padding_mask)

    out = []
    for each in prediction[0]:
        idx = np.argmax(each)
        if idx == zh_eos_flag:
            break
        out.append(idx)
        pass
    output_text = zh_tokenizer.sequences_to_texts(np.array([out]))
    print(output_text)
    print(en_tokenizer.sequences_to_texts(np.array([input_text[0]])))
    print()
