import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, UpSampling2D
import keras.models
import keras.regularizers


def res_basic_block(n_filters, stage=0, block=0, kernel_size=3, stride=None):
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    def f(x):
        y = Conv2D(n_filters, kernel_size, strides=stride, use_bias=False, padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = Conv2D(n_filters, kernel_size, use_bias=False, padding='same')(y)
        y = BatchNormalization()(y)

        if block == 0:
            shortcut = Conv2D(n_filters, 1, strides=stride, use_bias=False)(x)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = x

        y = Add()([y, shortcut])
        return Activation("relu")(y)

    return f


def res_bottleneck_block(n_filters, stage=0, block=0, kernel_size=3, stride=None):
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    def f(x):
        y = Conv2D(n_filters, 1, strides=stride, use_bias=False, padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = Conv2D(n_filters, kernel_size, use_bias=False, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = Conv2D(n_filters * 4, 1, use_bias=False)(y)
        y = BatchNormalization()(y)

        if block == 0:
            shortcut = Conv2D(n_filters * 4, 1, strides=stride, use_bias=False)(x)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = x

        y = Add()([y, shortcut])
        return Activation("relu")(y)

    return f


def resnet(inputs, blocks, block, n_init_features=64):

    x = Conv2D(n_init_features, 7, strides=2, use_bias=False, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(3, strides=2, padding="same")(x)

    features = n_init_features
    outputs = []
    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id)(x)
        features *= 2
        outputs.append(x)

    return outputs


def FPN(inputs, blocks, block, n_init_features=64):

    outputs = resnet(inputs, blocks, block, n_init_features)
    c2, c3, c4, c5 = outputs

    p5 = Conv2D(256, 1, strides=1, padding="same")(c5)
    upsampled_p5 = UpSampling2D(interpolation="bilinear", size=2)(p5)

    p4 = Conv2D(256, 1, strides=1, padding="same")(c4)
    p4 = Add()([upsampled_p5, p4])
    upsampled_p4 = UpSampling2D(interpolation="bilinear", size=2)(p4)
    p4 = Conv2D(256, 3, strides=1, padding="same")(p4)

    p3 = Conv2D(256, 1, strides=1, padding="same")(c3)
    p3 = Add()([upsampled_p4, p3])
    upsampled_p3 = UpSampling2D(interpolation="bilinear", size=2)(p3)
    p3 = Conv2D(256, 3, strides=1, padding="same")(p3)

    p2 = Conv2D(256, 1, strides=1, padding="same")(c2)
    p2 = Add()([upsampled_p3, p2])
    p2 = Conv2D(256, 3, strides=1, padding="same")(p2)

    return [p2, p3, p4, p5]


def resnet18(inputs, n_init_features=64):
    return resnet(inputs, [2, 2, 2, 2], res_basic_block, n_init_features)


def resnet34(inputs, n_init_features=64):
    return resnet(inputs, [3, 4, 6, 3], res_basic_block, n_init_features)


def resnet50(inputs, n_init_features=64):
    return resnet(inputs, [3, 4, 6, 3], res_bottleneck_block, n_init_features)


def resnet101(inputs, n_init_features=64):
    return resnet(inputs, [3, 4, 23, 3], res_bottleneck_block, n_init_features)


def resnet152(inputs, n_init_features=64):
    return resnet(inputs, [3, 8, 36, 3], res_bottleneck_block, n_init_features)


def resnet200(inputs, n_init_features=64):
    return resnet(inputs, [3, 24, 36, 3], res_bottleneck_block, n_init_features)


def FPN18(inputs, n_init_features=64):
    return FPN(inputs, [2, 2, 2, 2], res_basic_block, n_init_features)


def FPN34(inputs, n_init_features=64):
    return FPN(inputs, [3, 4, 6, 3], res_basic_block, n_init_features)


def FPN50(inputs, n_init_features=64):
    return FPN(inputs, [3, 4, 6, 3], res_bottleneck_block, n_init_features)
