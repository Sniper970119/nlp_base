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
from sklearn.model_selection import train_test_split

from generate.utils import load_data, add_BOS_EOS
from generate.models.seq2seq import MyModel

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()

parser.add_argument('--model', default="seq2seq", type=str, help='choose a model: Seq2Seq')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model
    struct_file = import_module('models.' + model_name)
    config = struct_file.Config()

    input_tensor, target_tensor, tokenizer = load_data(config.train_path, config.num_samples)

    train_x, val_x, train_y, val_y = train_test_split(input_tensor, target_tensor, test_size=0.2)

    config.steps_per_epoch = len(train_x) // config.batch_size

    vocab_input_size = len(tokenizer.word_index) + 1
    vocab_target_size = len(tokenizer.word_index) + 1
    config.num_encoder_tokens = vocab_input_size
    config.num_decoder_tokens = vocab_target_size

    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x))
    dataset = dataset.batch(config.batch_size, drop_remainder=True)

    model = struct_file.MyModel(config)

    encoder, decoder = model.my_model()

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder
                                     )


    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = loss_obj(real, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_mean(loss)
        return loss


    def one_step(input, target, encoder_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            encoder_output, encoder_hidden = encoder(input, encoder_hidden)
            decoder_hidden = encoder_hidden
            decoder_input = tf.expand_dims([tokenizer.word_index['bos']] * config.batch_size, 1)

            for t in range(1, target.shape[1]):
                predictions, dec_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)

                loss += loss_function(target[:, t], predictions)
                x = target[:, t]
                decoder_input = tf.expand_dims(x, 1)

        batch_loss = (loss / int(target.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


    for epoch in range(config.num_epochs):
        import time
        start = time.time()

        encoder_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (input, targ)) in enumerate(dataset.take(config.steps_per_epoch)):
            batch_loss = one_step(input, targ, encoder_hidden)
            total_loss += batch_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=config.save_path)

        print('Epoch {} Loss {}'.format(epoch + 1,# :.8f
                                            total_loss / config.steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
