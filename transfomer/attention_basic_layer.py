from keras.layers import Dense, Layer, Reshape, Activation, Dot
import tensorflow as tf
import keras.backend as K


def scaled_attention(q, k, v, mask):
    k = tf.transpose(k, perm=[0, 1, 3, 2])
    qk = Dot(axes=[3, 2])([q, k])
    qk /= K.sqrt(K.cast(K.shape(q)[-1], tf.float32))
    # TODO implement masking here
    if mask is not None:
        qk = tf.add(qk, mask * -1e10)
    qk = Activation('softmax')(qk)
    qkv = Dot(axes=[3, 2])([qk, v])
    return tf.transpose(qkv, perm=[0, 2, 1, 3])


class QKVProjectionLayer(Layer):
    def __init__(self, embedding_size):
        super(QKVProjectionLayer).__init__()
        self.embedding_size = embedding_size
        self.wq = Dense(embedding_size)
        self.wk = Dense(embedding_size)
        self.wv = Dense(embedding_size)

    def __call__(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        return q, k, v


class MultiHeadAttention(Layer):
    def __init__(self, h, embedding_size):
        super(MultiHeadAttention).__init__()
        self.n_heads = h
        self.embedding_size = embedding_size
        self.d_k = self.embedding_size // self.n_heads
        self.wq = Dense(self.embedding_size)
        self.wk = Dense(self.embedding_size)
        self.wv = Dense(self.embedding_size)
        self.final_dense = Dense(self.embedding_size)

    def __call__(self, q, k, v, mask):
        batch_size = q.shape[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = Reshape((-1, self.n_heads, self.d_k))(q)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = Reshape((-1, self.n_heads, self.d_k))(k)
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = Reshape((-1, self.n_heads, self.d_k))(v)
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        attention = scaled_attention(q, k, v, mask)
        concat = Reshape((-1, self.embedding_size))(attention)
        output = self.final_dense(concat)
        return output


class PointwiseFeedForward(Layer):
    def __init__(self, ff_dim, embedding_size):
        super(PointwiseFeedForward).__init__()
        self.dense1 = Dense(ff_dim, activation='relu')
        self.dense2 = Dense(embedding_size)

    def __call__(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x
