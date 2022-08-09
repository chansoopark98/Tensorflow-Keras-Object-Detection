from .EfficientNet_lite import EfficientNetLiteB0
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Activation, ZeroPadding2D
from tensorflow.keras.layers import Input, Rescaling

class EfficientLiteB0():
    def __init__(self, image_size: tuple, pretrained: str = "imagenet", include_preprocessing: bool = False):
        self.image_size = image_size
        self.pretrained = pretrained
        self.include_preprocessing = include_preprocessing
    
    def build_backbone(self):
        if self.include_preprocessing:
            input_tensor = Input(shape=(*self.image_size, 3))
            input_tensor = Rescaling(1./255, offset=0.0,)(input_tensor)
        else:
            input_tensor = None

        base = EfficientNetLiteB0(include_top=False, weights=self.pretrained, input_shape=(*self.image_size, 3), input_tensor=input_tensor)

        self.kernel_initializer = {
            "class_name": "VarianceScaling",
            "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
        }

        return base

    def build_extra_layer(self):
        base = self.build_backbone()
        model_input = base.input

        base_channel = 64
        
        x2 = base.get_layer('block3b_add').output # 38x38 @ 48
        x3 = base.get_layer('block5c_add').output # 19x19 @ 112
        x4 = base.get_layer('block7a_project_bn').output #10x10 @  block6d_add 192 / block7a_project_bn 320
        
        # 5, 5
        x5 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same',
                    use_bias=False, kernel_initializer=self.kernel_initializer)(x4)
        x5 = BatchNormalization()(x5)
        x5 = ReLU(max_value=6)(x5)
        
        x5 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='x5_padding')(x5)
        x5 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=2, padding='valid', use_bias=False,
                             depthwise_initializer=self.kernel_initializer, pointwise_initializer=self.kernel_initializer)(x5)
        x5 = BatchNormalization()(x5)
        x5 = ReLU(max_value=6)(x5)
        
        # 3, 3
        x6 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                    kernel_initializer=self.kernel_initializer)(x5)
        x6 = BatchNormalization()(x6)
        x6 = ReLU(max_value=6)(x6)
        
        x6 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=False,
                             depthwise_initializer=self.kernel_initializer, pointwise_initializer=self.kernel_initializer)(x6)
        x6 = BatchNormalization()(x6)
        x6 = ReLU(max_value=6)(x6)
        
        # 1, 1
        x7 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same',
                    use_bias=False, kernel_initializer=self.kernel_initializer)(x6)
        x7 = BatchNormalization()(x7)
        x7 = ReLU(max_value=6)(x7)
        
        x7 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=False,
                             depthwise_initializer=self.kernel_initializer, pointwise_initializer=self.kernel_initializer)(x7)
        x7 = BatchNormalization()(x7)
        x7 = ReLU(max_value=6)(x7)
        

        features = [x2, x3, x4, x5, x6, x7]

        return model_input, features


if __name__ == '__main__':
    base = EfficientLiteB0(image_size=(300, 300))
    model = base.build_backbone()
    model.summary()