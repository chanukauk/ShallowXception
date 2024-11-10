import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Add, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D, Dense
from tensorflow.keras import Model

def conv_block(x, filters, block_num, conv_num, kernel_size, strides=1):
    name = f'block{block_num}_conv{conv_num}_'
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=name)(x)
    x = BatchNormalization(name=name + 'bn')(x)
    return x

def separable_conv_block(x, filters, block_num, conv_num, kernel_size, strides=1):
    name = f'block{block_num}_sepconv{conv_num}_'
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=name)(x)
    x = BatchNormalization(name=name + 'bn')(x)
    return x

def entry_flow(img_input):
    x = conv_block(img_input, filters=32, block_num='1', conv_num='1', kernel_size=3, strides=2)
    x = ReLU(name='block1_conv1_relu')(x)
    x = conv_block(x, filters=64, block_num='1', conv_num='2', kernel_size=3)
    x = ReLU(name='block1_conv2_relu')(x)

    residual = Conv2D(filters=128, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = separable_conv_block(x, filters=128, block_num='2', conv_num='1', kernel_size=3)
    x = ReLU(name='block2_sepconv2_relu')(x)
    x = separable_conv_block(x, filters=128, block_num='2', conv_num='2', kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='block2_pool')(x)
    x = Add()([x, residual])

    residual = Conv2D(filters=256, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = ReLU(name='block3_sepconv1_relu')(x)
    x = separable_conv_block(x, filters=256, block_num='3', conv_num='1', kernel_size=3)
    x = ReLU(name='block3_sepconv2_relu')(x)
    x = separable_conv_block(x, filters=256, block_num='3', conv_num='2', kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='block3_pool')(x)
    x = Add()([x, residual])

    residual = Conv2D(filters=728, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = ReLU(name='block4_sepconv1_relu')(x)
    x = separable_conv_block(x, filters=728, block_num='4', conv_num='1', kernel_size=3)
    x = ReLU(name='block4_sepconv2_relu')(x)
    x = separable_conv_block(x, filters=728, block_num='4', conv_num='2', kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='block4_pool')(x)
    x = Add()([x, residual])

    return x

def middle_flow(x, depth=8):
    block_num = 5
    for _ in range(depth):
        residual = x
        x = ReLU(name=f'block{block_num}_sepconv1_relu')(x)
        x = separable_conv_block(x, filters=728, block_num=block_num, conv_num='1', kernel_size=3)
        x = ReLU(name=f'block{block_num}_sepconv2_relu')(x)
        x = separable_conv_block(x, filters=728, block_num=block_num, conv_num='2', kernel_size=3)
        x = ReLU(name=f'block{block_num}_sepconv3_relu')(x)
        x = separable_conv_block(x, filters=728, block_num=block_num, conv_num='3', kernel_size=3)
        x = Add()([x, residual])
        block_num += 1
    return x

def exit_flow(x, num_classes):
    residual = Conv2D(filters=1024, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = ReLU(name='block13_sepconv1_relu')(x)
    x = separable_conv_block(x, filters=728, block_num=13, conv_num='1', kernel_size=3)
    x = ReLU(name='block13_sepconv2_relu')(x)
    x = separable_conv_block(x, filters=1024, block_num=13, conv_num='2', kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name='block13_pool')(x)
    x = Add()([x, residual])

    x = separable_conv_block(x, filters=1536, block_num=14, conv_num='1', kernel_size=3)
    x = ReLU(name='block14_sepconv1_relu')(x)
    x = separable_conv_block(x, filters=2048, block_num=14, conv_num='2', kernel_size=3)
    x = ReLU(name='block14_sepconv2_relu')(x)

    x = GlobalAvgPool2D(name='avg_pool')(x)
    x = Dense(units=num_classes, activation='softmax', name='predictions')(x)
    return x

def build_model(H, W, num_classes, depth=8):
    input_img = Input(shape=(H, W, 3))
    x = entry_flow(input_img)
    x = middle_flow(x, depth)
    output = exit_flow(x, num_classes)
    model = Model(inputs=input_img, outputs=output)
    return model
