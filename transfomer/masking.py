import tensorflow as tf
import numpy as np


def look_ahead_mask(shape):
    mask = tf.linalg.band_part(tf.ones((shape, shape)), 0, -1) - tf.linalg.band_part(tf.ones((shape, shape)), 0, 0)
    return mask


def padding_mask(array_to_pad, padding_token=0):
    mask = np.where(array_to_pad == padding_token, 1, 0)
    return mask[:, np.newaxis, np.newaxis, :]
