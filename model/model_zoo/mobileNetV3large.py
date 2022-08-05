from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D

class MobileNetV3L():
    def __init__(self, image_size: tuple, pretrained: str = "imagenet"):
        self.image_size = image_size
        self.pretrained = pretrained
    
    def build_backbone(self):
        base = MobileNetV3Large(input_shape=[*self.image_size, 3], include_preprocessing=True, include_top=False)
        
        return base

    def build_extra_layer(self):
        base = self.build_backbone()
        model_input = base.input

        base_channel = 128

        x2 = base.get_layer('expanded_conv_5/Add').output # 38x38 @ 40
        x3 = base.get_layer('expanded_conv_11/Add').output # 19x19 @ 112
        x4 = base.get_layer('expanded_conv_14/Add').output # 10x10 @ 160

        x5 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=False)(x4)
        x5 = BatchNormalization()(x5)
        x5 = ReLU(6.)(x5)
        

        x5 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=2, padding='same', use_bias=False)(x5)
        x5 = BatchNormalization()(x5)
        x5 = ReLU(6.)(x5)
        

        x6 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=False)(x5)
        x6 = BatchNormalization()(x6)
        x6 = ReLU(6.)(x6)
        
        x6 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=False)(x6)
        x6 = BatchNormalization()(x6)
        x6 = ReLU(6.)(x6)
        

        x7 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=False)(x6)
        x7 = BatchNormalization()(x7)
        x7 = ReLU(6.)(x7)

        x7 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=False)(x7)
        x7 = BatchNormalization()(x7)
        x7 = ReLU(6.)(x7)
        

        features = [x2, x3, x4, x5, x6, x7]

        return model_input, features


if __name__ == '__main__':
    base = MobileNetV3L(image_size=(300, 300))
    model = base.build_backbone()
    model.summary()
    