from keras.losses import sparse_categorical_crossentropy
import numpy as np
import keras.backend as K
import tensorflow as tf


def transformer_schedule(d_model, warmup_steps):
    def schedule(epoch, lr):
        lr = np.sqrt(d_model) * np.minimum(np.sqrt(epoch), epoch * np.power(warmup_steps, -1.5))
        return lr
    return schedule


def masked_loss(y_true, y_pred):
    masked_true = K.not_equal(y_true, 0)
    masked_true = K.cast(masked_true, tf.float32)
    loss = sparse_categorical_crossentropy(y_true, y_pred)
    masked_loss = masked_true * loss
    return K.mean(masked_loss)



