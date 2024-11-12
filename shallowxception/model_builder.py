import re
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D
from tensorflow.keras import Model
from tensorflow.keras.applications import Xception

class ShallowXceptionModel:
    def __init__(self, img_height, img_width, num_classes, depth=8):
        
        if depth > 8:
            raise ValueError("Depth must be 8 or less.")
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.depth = depth

    def conv_block(self, x, filters, block_num, conv_num, kernel_size, strides=1):
        name = f'block{block_num}_conv{conv_num}_'
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=name)(x)
        x = BatchNormalization(name=name+'bn')(x)
        return x

    def separable_conv_block(self, x, filters, block_num, conv_num, kernel_size, strides=1):
        name = f'block{block_num}_sepconv{conv_num}_'
        x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=name)(x)
        x = BatchNormalization(name=name+'bn')(x)
        return x

    def entry_flow(self, img_input):
        # Block 1
        x = self.conv_block(img_input, filters=32, block_num=1, conv_num=1, kernel_size=3, strides=2)
        x = ReLU(name='block1_conv1_relu')(x)
        x = self.conv_block(x, filters=64, block_num=1, conv_num=2, kernel_size=3)
        x = ReLU(name='block1_conv2_relu')(x)
        
        # Block 2
        residual = Conv2D(filters=128, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        
        x = self.separable_conv_block(x, filters=128, block_num=2, conv_num=1, kernel_size=3)
        x = ReLU(name='block2_sepconv2_relu')(x)
        x = self.separable_conv_block(x, filters=128, block_num=2, conv_num=2, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same', name='block2_pool')(x)
        x = Add()([x, residual])

        # Block 3
        residual = Conv2D(filters=256, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        
        x = ReLU(name='block3_sepconv1_relu')(x)
        x = self.separable_conv_block(x, filters=256, block_num=3, conv_num=1, kernel_size=3)
        x = ReLU(name='block3_sepconv2_relu')(x)
        x = self.separable_conv_block(x, filters=256, block_num=3, conv_num=2, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same', name='block3_pool')(x)
        x = Add()([x, residual])

        # Block 4
        residual = Conv2D(filters=728, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        
        x = ReLU(name='block4_sepconv1_relu')(x)
        x = self.separable_conv_block(x, filters=728, block_num=4, conv_num=1, kernel_size=3)
        x = ReLU(name='block4_sepconv2_relu')(x)
        x = self.separable_conv_block(x, filters=728, block_num=4, conv_num=2, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same', name='block4_pool')(x)
        x = Add()([x, residual])
        
        return x

    def middle_flow(self, tensor):
        block_num = 5
        for _ in range(self.depth):
            x = ReLU(name=f'block{block_num}_sepconv1_relu')(tensor)
            x = self.separable_conv_block(x, filters=728, block_num=block_num, conv_num=1, kernel_size=3)

            x = ReLU(name=f'block{block_num}_sepconv2_relu')(x)
            x = self.separable_conv_block(x, filters=728, block_num=block_num, conv_num=2, kernel_size=3)

            x = ReLU(name=f'block{block_num}_sepconv3_relu')(x)
            x = self.separable_conv_block(x, filters=728, block_num=block_num, conv_num=3, kernel_size=3)
            tensor = Add()([tensor, x])
            block_num += 1

        return tensor

    def exit_flow(self, tensor):
        block_num = self.depth + 5
        
        residual = Conv2D(filters=1024, kernel_size=1, strides=2,  padding='same', use_bias=False)(tensor)
        residual = BatchNormalization()(residual)
        
        x = ReLU(name=f'block{block_num}_sepconv1_relu')(tensor)
        x = self.separable_conv_block(x, filters=728, block_num=block_num, conv_num=1, kernel_size=3)

        x = ReLU(name=f'block{block_num}_sepconv2_relu')(x)
        x = self.separable_conv_block(x, filters=1024, block_num=block_num, conv_num=2, kernel_size=3)

        x = MaxPool2D(pool_size=3, strides=2, padding='same', name=f'block{block_num}_pool')(x)
        x = Add()([x, residual])

        x = self.separable_conv_block(x, filters=1536, block_num=block_num+1, conv_num=1, kernel_size=3)
        x = ReLU(name=f'block{block_num+1}_sepconv1_relu')(x)

        x = self.separable_conv_block(x, filters=2048, block_num=block_num+1, conv_num=2, kernel_size=3)
        x = ReLU(name=f'block{block_num+1}_sepconv2_relu')(x)

        x = GlobalAvgPool2D(name='avg_pool')(x)
        x = Dense(units=self.num_classes, activation='softmax', name='predictions')(x)

        return x

    def build_model(self):
        input_img = Input(shape=(self.img_height, self.img_width, 3))
        x = self.entry_flow(input_img)
        x = self.middle_flow(x)
        output = self.exit_flow(x)

        model = Model(inputs=input_img, outputs=output)
        return model

    def pre_trained_weights_transfer(self, model):
        depth_block_map = ['block5', 'block6',  'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
        patterns = depth_block_map[self.depth:]

        Xception_model = Xception(weights='imagenet', include_top=False)

        custom_weights = model.get_weights()
        not_include = False
        extracted_weights = []

        for layer in Xception_model.layers:
            for pattern in patterns:
                if re.search(pattern, layer.name):
                    not_include = True
                    break
            if not_include is False:
                extracted_weights_list = layer.get_weights()
                for extracted_weight in extracted_weights_list:
                    extracted_weights.append(extracted_weight)

            not_include = False

        custom_weights[:-2] = extracted_weights
        model.set_weights(custom_weights)

        return model
