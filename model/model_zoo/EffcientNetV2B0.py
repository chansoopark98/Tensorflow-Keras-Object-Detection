from tensorflow.keras import  Input
from .EfficientNetV2 import EfficientNetV2B0 as EffV2B0
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Activation

class EfficientNetV2B0():
    def __init__(self, image_size: tuple, pretrained: str = "imagenet"):
        self.image_size = image_size
        self.pretrained = pretrained
    
    def build_backbone(self):
        base = EffV2B0(input_shape=(*self.image_size, 3), first_strides=2, num_classes=0, pretrained=self.pretrained)
        # base = efficientnet_v2.EfficientNetV2B0(weights=self.pretrained, include_top=False,
        #                                         input_shape=[*self.image_size, 3], input_tensor=input_tensor,
        #                          include_preprocessing=False)
        return base

    def build_extra_layer(self):
        base = self.build_backbone()
        model_input = base.input

        base_channel = 64

        x2 = base.get_layer('add_1').output # 38x38 @ 48
        x3 = base.get_layer('add_7').output # 19x19 @ 112
        x4 = base.get_layer('add_14').output # 10x10 192
        
        # 5, 5
        x5 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=False)(x4)
        x5 = BatchNormalization(momentum=0.9)(x5)
        x5 = Activation('swish')(x5)
        
        x5 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=2, padding='same', use_bias=False)(x5)
        x5 = BatchNormalization(momentum=0.9)(x5)
        x5 = Activation('swish')(x5)
        
        # 3, 3
        x6 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=False)(x5)
        x6 = BatchNormalization(momentum=0.9)(x6)
        x6 = Activation('swish')(x6)
        
        
        x6 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=False)(x6)
        x6 = BatchNormalization(momentum=0.9)(x6)
        x6 = Activation('swish')(x6)
        
        # 1, 1
        x7 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=False)(x6)
        x7 = BatchNormalization(momentum=0.9)(x7)
        x7 = Activation('swish')(x7)
        
        
        x7 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=False)(x7)
        x7 = BatchNormalization(momentum=0.9)(x7)
        x7 = Activation('swish')(x7)
        

        features = [x2, x3, x4, x5, x6, x7]

        return model_input, features


if __name__ == '__main__':
    base = EfficientNetV2B0(image_size=(300, 300))
    model = base.build_backbone()
    model.summary()