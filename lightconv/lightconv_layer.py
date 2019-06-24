import keras.backend as K
from keras.layers import Dense, Layer, DepthwiseConv2D, Reshape, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import tensorflow as tf

from lightconv.softmax_constraint import SoftmaxConstraint


def LightConvBlock(x, d, H, k=3):
    assert d % 2 == 0, 'Vector dimensionality should be even to allow GLU splitting'
    x = Dense(2 * d)(x)
    x = glu(x, dim=-1)
    x = Reshape((-1, H, d // H))(x)
    constraint = SoftmaxConstraint()
    x = DepthwiseConv2D(kernel_size=(1, k), kernel_constraint=SoftmaxConstraint(), padding='same')(x)
    x = Dense(d)(x)
    return x


def glu(x, dim=-1):
    a, b = tf.split(x, 2, dim)
    b = K.sigmoid(b)
    return a * b


class GLU(Layer):
    def __init__(self, dim=-1):
        super(GLU).__init__()
        self.dim = dim

    def __call__(self, x):

        return glu(x, self.dim)


class LightConv(Layer):
    def __init__(self, k_size, H, d, glu_split_dim=-1):
        super(LightConv).__init__()
        self.H = H
        self.d = d
        self.glu_split_dim = glu_split_dim
        self.input_dense = Dense(2 * d)
        self.dep_conv = DepthwiseConv2D(kernel_size=(k_size, 1), kernel_constraint=SoftmaxConstraint(), padding='same')
        self.output_dense = Dense(d)

    def __call__(self, x):
        x = self.input_dense(x)
        #x = Lambda(lambda i: glu(i, self.glu_split_dim))(x) #if you need to compile it as keras model
        x = glu(x)
        x = Reshape((-1, self.d // self.H, self.H))(x)
        x = self.dep_conv(x)
        x = Reshape((-1, self.d))(x)
        x = self.output_dense(x)
        return x


if __name__ == '__main__':
    d = 1024
    H = 16
    k = 6
    input = tf.zeros((3, 100, d))
    output = LightConvBlock(input, d, H, k)

    output_layer = LightConv(k_size=k, H=H, d=d)

    input = Input((100, d))
    output = output_layer(input)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=categorical_crossentropy)
