# coding=utf-8

from keras.layers import Embedding, Layer, Dropout, Dense, RepeatVector
import keras.backend as K
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import numpy as np
import tensorflow as tf
from transfomer.attention_blocks import EncoderAttentionBlock, DecoderAttentionBlock
from transfomer.training_utils import transformer_schedule
from transfomer.masking import look_ahead_mask, padding_mask
from lightconv.lightconv_layer import LightConv
import tensorflow_datasets as tfds


def prepare_tokenizer(sequences):
    tokenizer = Tokenizer(oov_token='OOV')  # TODO Consider changing to subword encoding
    texts = ['SOS ' + seq.numpy().decode('unicode_escape').encode('ISO-8859-1').decode('utf-8') + ' EOS' for seq in
             sequences]
    tokenizer.fit_on_texts(texts=texts)
    tokenized_texts = tokenizer.texts_to_sequences(texts)
    return tokenizer, tokenized_texts


def filter_pad(tokenized_lang1, tokenized_lang2, max_len=20):
    filtered_lang1, filtered_lang2 = zip(*[(seq_pt, seq_eng) for seq_eng, seq_pt in zip(tokenized_lang1,
                                                                                        tokenized_lang2)
                                         if (len(seq_pt) < max_len and len(seq_eng) < max_len)])
    padded_lang1 = pad_sequences(filtered_lang1, padding='post')
    padded_lang2 = pad_sequences(filtered_lang2, padding='post')
    return padded_lang1, padded_lang2


class PositionalEncoding(Layer):
    def __init__(self, embedding_size):
        super(PositionalEncoding).__init__()
        self.embedding_size = embedding_size

    def __call__(self, x):
        return self.pos(x)

    def pos(self, vec):
        pos = np.arange(vec.shape[1].value)[:, np.newaxis]
        batch_size = vec.shape[0].value
        i = np.arange(self.embedding_size)[np.newaxis, :]
        angle = pos / (np.power(10000, 2 * i / self.embedding_size))
        sines = np.sin(angle[:, 0::2])
        cosines = np.cos(angle[:, 1::2])
        output = np.empty_like(angle)
        output[:, 0::2] = sines
        output[:, 1::2] = cosines
        output = RepeatVector(batch_size)(tf.convert_to_tensor(output, dtype=tf.float32))
        return K.permute_dimensions(output, (1,0, 2))


class Encoder(Layer):

    def __init__(self, input_dim, output_dim, n_blocks=6, rate=0.2, ff_dim=128, n_heads=4,
                 attention_type='multihead-self'):
        super(Encoder).__init__()
        self.rate = rate
        self.embedding = Embedding(input_dim, output_dim)
        self.positional_encoding = PositionalEncoding(output_dim)
        self.output_dim = output_dim
        self.attention_blocks = [EncoderAttentionBlock(n_heads, output_dim, rate=rate, ff_dim=ff_dim,
                                                       attention_type=attention_type) for _ in range(n_blocks)]

    def __call__(self, x, mask):
        x = self.embedding(x)
        x *= K.sqrt(tf.convert_to_tensor(self.output_dim, tf.float32))
        x += self.positional_encoding(x)
        x = Dropout(rate=self.rate)(x)
        for block in self.attention_blocks:
            x = block(x, mask)
        return x


class Decoder(Layer):
    def __init__(self, input_dim, output_dim, n_blocks=6, rate=0.2, ff_dim=128, n_heads=4,
                 attention_type='multihead-self'):
        super(Decoder).__init__()
        self.rate = rate
        self.embedding = Embedding(input_dim, output_dim)
        self.positional_encoding = PositionalEncoding(output_dim)
        self.output_dim = output_dim
        self.attention_blocks = [DecoderAttentionBlock(n_heads, output_dim, rate=rate, ff_dim=ff_dim,
                                                       attention_type=attention_type) for _ in range(n_blocks)]

    def __call__(self, x, encoder_output, mask1, mask2):
        x = self.embedding(x)
        x *= K.sqrt(tf.convert_to_tensor(self.output_dim, tf.float32))
        x += self.positional_encoding(x)
        x = Dropout(rate=self.rate)(x)
        for block in self.attention_blocks:
            x = block(x, encoder_output, mask1, mask2)
        return x


class Transformer(Model):
    def __init__(self, input_vocab_size, output_vocab_size, n_blocks=4, rate=0.2, hidden_size=128, n_heads=8,
                 ff_dim=512, attention_type='multihead-self'):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_vocab_size, hidden_size, n_blocks=n_blocks,
                               rate=rate, n_heads=n_heads, ff_dim=ff_dim, attention_type=attention_type)
        self.decoder = Decoder(output_vocab_size, hidden_size, n_blocks=n_blocks,
                               rate=rate, n_heads=n_heads, ff_dim=ff_dim, attention_type=attention_type)
        self.final_dense = Dense(output_vocab_size, activation='softmax')

    def __call__(self, input, target, enc_mask, dec_mask, look_ahead_mask):
        enc_output = self.encoder(input, enc_mask)
        dec_output = self.decoder(target, enc_output, look_ahead_mask, dec_mask)
        output = self.final_dense(dec_output)
        return output



if __name__ == '__main__':
    tf.enable_eager_execution()
    examples, meta = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
    train, test = examples['train'], examples['validation']

    pt_sequences, en_sequences = zip(*train)
    tokenizer_en, tokenized_eng = prepare_tokenizer(en_sequences)
    tokenizer_pt, tokenized_pt = prepare_tokenizer(pt_sequences)

    padded_eng, padded_pt = filter_pad(tokenized_eng, tokenized_pt)
    #
    input_vocab_size = len(tokenizer_en.word_index)
    output_vocab_size = len(tokenizer_pt.word_index)
    # # transformer = Transformer(500, 40)
    # # output = transformer(input, target, None, None, None)

    #
    n_samples = 100
    decoder_input = padded_pt[:n_samples, :-1]
    target = padded_pt[:n_samples, 1:]
    encoder_input = padded_eng[:n_samples, :]

    enc_masking = padding_mask(encoder_input)
    dec_masking = padding_mask(encoder_input)
    look_ahead_masking = look_ahead_mask(decoder_input.shape[1])
    dec_target_masking = padding_mask(target)
    look_ahead_masking = np.array([np.maximum(dec_target_mask_row, look_ahead_masking) for dec_target_mask_row in dec_target_masking])

    optimizer = Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    schedule = transformer_schedule(d_model=128, warmup_steps=4000)
    scheduler = LearningRateScheduler(schedule)

    transformer = Transformer(input_vocab_size, output_vocab_size, hidden_size=1024, attention_type='lightconv')
    output = transformer(tf.convert_to_tensor(encoder_input), tf.convert_to_tensor(decoder_input), enc_masking,
                         dec_masking,
                         look_ahead_masking)
    print('Finished')
