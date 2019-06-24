from keras.optimizers import Adam
from keras.layers import BatchNormalization, Activation, Add, Concatenate,  Input
from keras.models import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, DepthwiseConv2D
from segmentation.deeplab_model import atrous_pyramid_layer, conv_norm_activate


def sep_conv_norm(x, n_filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1)):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, 1, padding='same')(x)
    # Should another norm be put here?
    return x


def xception_block(x, n_filters1, n_filters2, res_type='conv', dilation_rate=(1, 1), strides=(1, 1)):
    assert res_type in ['conv', 'plain', None]
    if res_type == 'conv':
        res = Conv2D(n_filters2, 1, strides=strides)(x)
    elif res_type == 'plain':
        res = x
    elif res_type is None:
        res = None
    # conv1 = SeparableConvolution2D(n_filters1, 3, padding='same')(input)
    # conv2 = SeparableConvolution2D(n_filters1, 3, padding='same')(conv1)
    # conv3 = SeparableConvolution2D(n_filters2, 3, strides=stride, padding='same')(conv2)
    conv1 = sep_conv_norm(x, n_filters1, 3, dilation_rate=dilation_rate)
    conv2 = sep_conv_norm(conv1, n_filters1, 3, dilation_rate=dilation_rate)
    conv3 = sep_conv_norm(conv2, n_filters2, 3, strides=strides, dilation_rate=dilation_rate)
    if res:
        output = Add()([res, conv3])
    else:
        output = conv3
    return output


def deeplab_3plus(n_classes, num_middle_blocks=16):
    # Input and initial convolutions
    input = Input((224, 224, 3), name='input_layer')
    init_conv = conv_norm_activate(input, 32, kernel_size=3, strides=2, padding='same')
    init_conv = conv_norm_activate(init_conv, 64, kernel_size=3, strides=1, padding='same')

    # Input flow Xception blocks
    xception_block_1 = xception_block(init_conv, 128, 128, strides=2)
    xception_block_2 = xception_block(xception_block_1, 256, 256, strides=2)
    xception_block_3 = xception_block(xception_block_2, 728, 728, strides=2)

    # Middle flow Xception blocks
    middle_block = xception_block_3
    for i in range(num_middle_blocks):
        middle_block = xception_block(middle_block, 728, 728, res_type='plain', strides=1)

    # Final blocks
    out_block_1 = xception_block(middle_block, 728, 1024, strides=1)
    out_block_2 = xception_block(out_block_1, 1536, 2048, res_type=None, dilation_rate=2, strides=1)

    # Pyramid
    pyramid = atrous_pyramid_layer(out_block_2, 14)
    pyramid = UpSampling2D(4, interpolation='bilinear')(pyramid)

    # Residual
    res = conv_norm_activate(xception_block_1, 256, kernel_size=1)
    concat = Concatenate()([pyramid, res])

    # Final convolutions
    final_conv = Conv2D(256, 3, padding='same')(concat)
    final_conv = Conv2D(256, 3, padding='same')(final_conv)
    output = Conv2D(n_classes, 1)(final_conv)
    output = UpSampling2D(4, interpolation='bilinear')(output)
    output = Activation('softmax')(output)

    model = Model(inputs=[input], outputs=[output])
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.summary()
    return model


deeplab_3plus_model = deeplab_3plus(2)
