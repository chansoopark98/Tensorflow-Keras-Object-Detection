from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D

class MobileNetV3S():
    def __init__(self, image_size: tuple, pretrained: str = "imagenet"):
        self.image_size = image_size
        self.pretrained = pretrained
    
    def build_backbone(self):
        base = MobileNetV3Small(input_shape=[*self.image_size, 3], include_preprocessing=False, include_top=False)
        
        return base

    def build_extra_layer(self):
        base = self.build_backbone()
        model_input = base.input

        base_channel = 64

        x2 = base.get_layer('expanded_conv_2/Add').output # 38x38 @ 192
        x3 = base.get_layer('expanded_conv_7/Add').output # 19x19 @ 576
        x4 = base.get_layer('expanded_conv_10/Add').output # 10x10 @ 160

        x5 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=True)(x4)
        x5 = BatchNormalization()(x5)
        x5 = ReLU(6.)(x5)
        

        x5 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=2, padding='same', use_bias=True)(x5)
        x5 = BatchNormalization()(x5)
        x5 = ReLU(6.)(x5)
        

        x6 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=True)(x5)
        x6 = BatchNormalization()(x6)
        x6 = ReLU(6.)(x6)
        
        x6 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=True)(x6)
        x6 = BatchNormalization()(x6)
        x6 = ReLU(6.)(x6)
        

        x7 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=True)(x6)
        x7 = BatchNormalization()(x7)
        x7 = ReLU(6.)(x7)

        x7 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=True)(x7)
        x7 = BatchNormalization()(x7)
        x7 = ReLU(6.)(x7)
        

        features = [x2, x3, x4, x5, x6, x7]

        return model_input, features


if __name__ == '__main__':
    base = MobileNetV3S(image_size=(300, 300))
    model = base.build_backbone()
    model.summary()
    