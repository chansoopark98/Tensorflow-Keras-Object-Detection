from tensorflow.keras import  Input
from .EfficientNetV2 import EfficientNetV2B0 as EffV2B0
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Activation, ZeroPadding2D
from tensorflow.keras import layers
from functools import reduce

class EfficientNetV2B0():
    def __init__(self, image_size: tuple, pretrained: str = "imagenet"):
        self.image_size = image_size
        self.pretrained = pretrained
        self.use_std_conv = False
        self.kernel_initializer = {
            "class_name": "VarianceScaling",
            "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
        }

        self.MOMENTUM = 0.997
        self.EPSILON = 1e-4
    
    def build_backbone(self):
        base = EffV2B0(input_shape=(*self.image_size, 3), first_strides=2, pretrained=self.pretrained, include_preprocessing=False, num_classes=0)
        return base

    def SeparableConvBlock(self, num_channels, kernel_size, strides, name, freeze_bn=False):
        f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                    use_bias=True, name=f'{name}/conv')
        f2 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON, name=f'{name}/bn')
        # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

    def build_BiFPN(self, features, num_channels, id, freeze_bn=False):
        if id == 0:
            C3, C4, C5 = features
            P3_in = C3
            P4_in = C4
            P5_in = C5
            P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
            P6_in = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON, name='resample_p6/bn')(P6_in)
            # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
            P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
            P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
            P7_U = layers.UpSampling2D()(P7_in)
            P6_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
            P6_td = layers.Activation('swish')(P6_td)
            P6_td = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
            P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
            P5_in_1 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
            # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
            P6_U = layers.UpSampling2D()(P6_td)
            P5_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
            P5_td = layers.Activation('swish')(P5_td)
            P5_td = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
            P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
            P4_in_1 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
            # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
            P5_U = layers.UpSampling2D()(P5_td)
            P4_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
            P4_td = layers.Activation('swish')(P4_td)
            P4_td = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
            P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
            P3_in = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
            # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
            P4_U = layers.UpSampling2D()(P4_td)
            P3_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
            P3_out = layers.Activation('swish')(P3_out)
            P3_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
            P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
            P4_in_2 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
            # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
            P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
            P4_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
            P4_out = layers.Activation('swish')(P4_out)
            P4_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

            P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
            P5_in_2 = layers.BatchNormalization(momentum=self.MOMENTUM, epsilon=self.EPSILON,
                                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
            # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
            P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
            P5_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
            P5_out = layers.Activation('swish')(P5_out)
            P5_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

            P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
            P6_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
            P6_out = layers.Activation('swish')(P6_out)
            P6_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

            P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
            P7_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
            P7_out = layers.Activation('swish')(P7_out)
            P7_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

        else:
            P3_in, P4_in, P5_in, P6_in, P7_in = features
            P7_U = layers.UpSampling2D()(P7_in)
            P6_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
            P6_td = layers.Activation('swish')(P6_td)
            P6_td = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
            P6_U = layers.UpSampling2D()(P6_td)
            P5_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
            P5_td = layers.Activation('swish')(P5_td)
            P5_td = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
            P5_U = layers.UpSampling2D()(P5_td)
            P4_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
            P4_td = layers.Activation('swish')(P4_td)
            P4_td = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
            P4_U = layers.UpSampling2D()(P4_td)
            P3_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
            P3_out = layers.Activation('swish')(P3_out)
            P3_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
            P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
            P4_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
            P4_out = layers.Activation('swish')(P4_out)
            P4_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

            P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
            P5_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
            P5_out = layers.Activation('swish')(P5_out)
            P5_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

            P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
            P6_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
            P6_out = layers.Activation('swish')(P6_out)
            P6_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

            P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
            P7_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
            P7_out = layers.Activation('swish')(P7_out)
            P7_out = self.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
        return P3_out, P4_td, P5_td, P6_td, P7_out

    def build_extra_layer(self):
        base = self.build_backbone()
        model_input = base.input

        base_channel = 128

        
        x2 = base.get_layer('add_1').output # 38x38 @ 48
        x3 = base.get_layer('add_7').output # 19x19 @ 112
        x4 = base.get_layer('post_swish').output #10x10 @ 192


        features = [x2, x3, x4]

        for i in range(2):
            features = self.build_BiFPN(features=features, num_channels=base_channel, id=i, freeze_bn=False)


        return model_input, features


if __name__ == '__main__':
    base = EfficientNetV2B0(image_size=(256, 256))
    _, _ = base.build_extra_layer()
    # model.summary()