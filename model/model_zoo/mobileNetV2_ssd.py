from tensorflow.keras import  Input
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D

class MobileNetV2():
    def __init__(self, image_size: tuple, pretrained: str = "imagenet"):
        self.image_size = image_size
        self.pretrained = pretrained
    
    def build_backbone(self):
        input_tensor = Input(shape=(*self.image_size, 3))
        base = mobilenet_v2.MobileNetV2(weights=self.pretrained, include_top=False,
                                        input_shape=[*self.image_size, 3], input_tensor=input_tensor)
        return base

    def build_extra_layer(self):
        base = self.build_backbone()
        model_input = base.input

        base_channel = 128

        x2 = base.get_layer('block_5_add').output # 38x38 @ 32
        x3 = base.get_layer('block_12_add').output # 19x19 @ 96
        x4 = base.get_layer('block_16_project_BN').output # 10x10 @ 320

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
    base = MobileNetV2(image_size=(300, 300))
    model = base.build_backbone()
