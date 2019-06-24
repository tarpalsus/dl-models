from keras.applications import resnet50
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Activation, Add, Concatenate, Lambda
from keras.models import Model
from keras.layers.convolutional import Conv2D, UpSampling2D
import keras.backend as K
import numpy as np


def mean(x):
    return K.mean(x, axis=[1, 2], keepdims=True)


def conv_norm_activate(input_layer, num_filters, kernel_size=(3, 3),
                       strides=(1, 1), activation='relu', dilation_rate=1, padding='same'):
    conv = Conv2D(num_filters, kernel_size, dilation_rate=dilation_rate, padding=padding, strides=strides)(input_layer)
    batch_norm = BatchNormalization()(conv)
    output = Activation(activation)(batch_norm)
    return output


def conv_block(input_layer, num_filters, identity=True, rate=2):
    conv1 = conv_norm_activate(input_layer, num_filters, dilation_rate=rate)
    conv2 = conv_norm_activate(conv1, num_filters, dilation_rate=rate)
    conv3 = conv_norm_activate(conv2, num_filters * 4, dilation_rate=rate, activation='linear')
    if identity:
        res = input_layer
    else:
        res = conv_norm_activate(input_layer, num_filters * 4, dilation_rate=rate, activation='linear')
    add = Add()([conv3, res])
    activation = Activation('relu')(add)
    return activation


def atrous_convolution_block(input_layer, num_filters=256, rate=2):
    conv_block1 = conv_block(input_layer, num_filters=num_filters, rate=rate, identity=False)
    conv_block1 = conv_block(conv_block1, num_filters=num_filters, rate=rate)
    conv_block1 = conv_block(conv_block1, num_filters=num_filters, rate=rate)
    return conv_block1


def atrous_pyramid_layer(input_layer, upsamling_factor=14):
    conv_1x1 = Conv2D(256, 1)(input_layer)
    conv_1x1 = BatchNormalization()(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)

    conv_3x3_1 = conv_norm_activate(input_layer, 256, activation='linear', dilation_rate=6)
    conv_3x3_2 = conv_norm_activate(input_layer, 256, activation='linear', dilation_rate=12)
    conv_3x3_3 = conv_norm_activate(input_layer, 256, activation='linear', dilation_rate=18)

    image_level_ft = Lambda(mean)(input_layer)
    image_level_ft = Conv2D(256, 1)(image_level_ft)
    image_level_ft = UpSampling2D(upsamling_factor)(image_level_ft)

    concat = Concatenate()([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_ft])
    concat = conv_norm_activate(concat, 256, kernel_size=1)
    return concat


def deeplab_v3(n_classes, use_atrous_block=True):
    resnet = resnet50.ResNet50(weights=None)
    resnet_crop_layer = resnet.get_layer('activation_40').output
    resnet_input = resnet.get_layer('input_1').input
    input_shape = resnet.get_layer('input_1').input_shape
    if use_atrous_block:
        pyramid_input = atrous_convolution_block(resnet_crop_layer, rate=2)
    else:
        pyramid_input = resnet_crop_layer
    # atrous_conv = atrous_convolution_block(atrous_conv, rate=4)
    # atrous_conv = atrous_convolution_block(atrous_conv, rate=8)
    # output = atrous_convolution_block(atrous_conv, rate=16)
    atrous_pyramid = atrous_pyramid_layer(pyramid_input)
    output = Conv2D(n_classes, 1)(atrous_pyramid)
    size = np.array(input_shape[1:3]) / 14
    output = UpSampling2D(size=size.astype(np.int32), interpolation='bilinear')(output)
    output = Activation('softmax')(output)
    optimizer = Adam()
    model = Model(inputs=[resnet_input], outputs=[output])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.summary()

    return model


if __name__ == '__main__':
    import numpy as np
    deeplab_model = deeplab_v3(2)
    test_input = np.zeros((500, 224, 224, 3))
    test_label = np.zeros((500, 224, 224, 2))
    deeplab_model.fit(test_input, test_label, batch_size=3)