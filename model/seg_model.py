from model.efficientnet_v2 import *
import tensorflow as tf
from model.resnet101 import ResNet
from tensorflow.keras import layers

activation = 'relu'

class GlobalAveragePooling2D(tf.keras.layers.GlobalAveragePooling2D):
    def __init__(self, keep_dims=False, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).call(inputs)
        else:
            return tf.keras.backend.mean(inputs, axis=[1, 2], keepdims=True)

    def compute_output_shape(self, input_shape):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).compute_output_shape(input_shape)
        else:
            input_shape = tf.TensorShape(input_shape).as_list()
            return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])

    def get_config(self):
        config = super(GlobalAveragePooling2D, self).get_config()
        config['keep_dim'] = self.keep_dims
        return config


class Concatenate(tf.keras.layers.Concatenate):
    def __init__(self, out_size=None, axis=-1, name=None):
        super(Concatenate, self).__init__(axis=axis, name=name)
        self.out_size = out_size

    def call(self, inputs):
        return tf.keras.backend.concatenate(inputs, self.axis)

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        if self.out_size is None:
            return super(Concatenate, self).compute_output_shape(input_shape)
        else:
            if not isinstance(input_shape, list):
                raise ValueError('A `Concatenate` layer should be called '
                                 'on a list of inputs.')
            input_shapes = input_shape
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                if output_shape[self.axis] is None or shape[self.axis] is None:
                    output_shape[self.axis] = None
                    break
                output_shape[self.axis] += shape[self.axis]
            return tuple([output_shape[0]] + list(self.out_size) + [output_shape[-1]])

    def get_config(self):
        config = super(Concatenate, self).get_config()
        config['out_size'] = self.out_size
        return config




def csnet_seg_model(weights='pascal_voc', input_tensor=None, input_shape=(512, 1024, 3), classes=20, OS=16):
    # if not (weights in {'pascal_voc', None}):
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization) or `pascal_voc` '
    #                      '(pre-trained on PASCAL VOC)')
    #
    # if K.backend() != 'tensorflow':
    #     raise RuntimeError('The Deeplabv3+ model is only available with '
    #                        'the TensorFlow backend.')
    #
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    #
    #
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    #
    # x = Conv2D(32, (3, 3), strides=(2, 2),
    #            name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    # x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    # x = Activation('relu')(x)
    #
    # x = conv2d_same(x, 32, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    # x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    # x = Activation('relu')(x)
    #
    # x = xception_block(x, [128, 128, 128], 'entry_flow_block1',
    #                    skip_connection_type='conv', stride=2,
    #                    depth_activation=False)
    # x, skip1 = xception_block(x, [256, 256, 256], 'entry_flow_block2',
    #                           skip_connection_type='conv', stride=2,
    #                           depth_activation=False, return_skip=True)
    #
    # x = xception_block(x, [728, 728, 728], 'entry_flow_block3',
    #                    skip_connection_type='conv', stride=entry_block3_stride,
    #                    depth_activation=False)
    # for i in range(16):
    #     x = xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
    #                        skip_connection_type='sum', stride=1, rate=middle_block_rate,
    #                        depth_activation=False)
    #
    # x = xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
    #                    skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
    #                    depth_activation=False)
    # x = xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
    #                    skip_connection_type='none', stride=1, rate=exit_block_rates[1],
    #                    depth_activation=True)

    input_tensor = tf.keras.Input(shape=(512, 1024, 3))
    encoder = ResNet('ResNet101', [1, 2])
    c2, c5 = encoder(input_tensor, ['c2', 'c5'])
    aspp_size = (512 // 16, 1024 // 16)

    # """ for resnet101 """
    # base = resnet101.ResNet101(include_top=False, input_shape=input_shape, weights='imagenet')
    # base = ResNet101(include_top=False, input_shape=input_shape, weights='imagenet')
    # base.summary()
    divide_output_stride = 4
    # # x = base.get_layer('conv4_block23_out').output
    # x = base.get_layer('conv5_block3_out').output
    # # skip1 = base.get_layer('conv2_block3_out').output
    # skip1 = base.get_layer('conv2_block3_out').output
    # # conv5_block3_out 16, 32, 2048
    # # conv3_block4_out 32, 128, 512

    """ for EfficientNetV2S """


    # # efficientnetv2 small
    # divide_output_stride = 4
    # base = EfficientNetV2S(input_shape=input_shape, classifier_activation=None, survivals=None, dropout=0)
    # base.summary()
    # base.load_weights('./checkpoints/efficientnetv2-s-21k-ft1k.h5', by_name=True)
    # # x = base.get_layer('add_34').output # 32x64
    # x = base.get_layer('post_relu').output # 32x64
    # # skip1 = base.get_layer('stack_3_block0_sortcut_relu').output # 128x256
    # skip1 = base.get_layer('add_4').output # 128x256

    # efficientnetv2 medium
    # base = EfficientNetV2('m', input_shape=input_shape, classifier_activation=None, first_strides=1)
    # base.load_weights('checkpoints/efficientnetv2-m-21k-ft1k.h5', by_name=True)
    # x = base.get_layer('add_50').output # 32x64
    # skip1 = base.get_layer('add_9').output # 128x256

    x = _aspp(c5, 256)
    x = layers.Dropout(rate=0.5)(x)

    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = _conv_bn_relu(x, 48, 1, strides=1)

    x = Concatenate(out_size=aspp_size)([x, c2])
    x = _conv_bn_relu(x, 256, 3, 1)
    x = layers.Dropout(rate=0.5)(x)

    x = _conv_bn_relu(x, 256, 3, 1)
    x = layers.Dropout(rate=0.1)(x)

    x = layers.Conv2D(20, 1, strides=1)(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    return input_tensor, x


def _conv_bn_relu(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def _aspp(x, out_filters):
    xs = list()
    x1 = layers.Conv2D(out_filters, 1, strides=1)(x)
    xs.append(x1)
    aspp_size = (512//16, 1024//16)
    for i in range(3):
        xi = layers.Conv2D(out_filters, 3, strides=1, padding='same', dilation_rate=6 * (i + 1))(x)
        xs.append(xi)
    img_pool = GlobalAveragePooling2D(keep_dims=True)(x)
    img_pool = layers.Conv2D(out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
    img_pool = layers.UpSampling2D(size=aspp_size, interpolation='bilinear')(img_pool)
    xs.append(img_pool)

    x = Concatenate(out_size=aspp_size)(xs)
    x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    return x

    """ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    #
    # # end of feature extractor
    #
    # # branching for Atrous Spatial Pyramid Pooling
    # # simple 1x1
    # b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    # b0 = Activation(activation, name='aspp0_activation')(b0) # 16, 32 @ 256
    #
    # # rate = 6 (12)
    # b1 = SepConv_BN(x, 256, 'aspp1',
    #                 rate=atrous_rates[0], depth_activation=True, epsilon=1e-5) # 16, 32 @ 256
    # # rate = 12 (24)
    # b2 = SepConv_BN(x, 256, 'aspp2',
    #                 rate=atrous_rates[1], depth_activation=True, epsilon=1e-5) # 16, 32 @256
    # # rate = 18 (36)
    # b3 = SepConv_BN(x, 256, 'aspp3',
    #                 rate=atrous_rates[2], depth_activation=True, epsilon=1e-5) # 16, 32 @256
    #
    # # Image Feature branch
    # out_shape = int(np.ceil(input_shape[0] / OS)) # os16 => 32
    # b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x) # 1,2
    # b4 = Conv2D(256, (1, 1), padding='same',
    #             use_bias=False, name='image_pooling')(b4)
    # b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    # b4 = Activation(activation)(b4)
    #
    # b4 = BilinearUpsampling((out_shape, out_shape))(b4) # 16, 32
    #
    #
    # # concatenate ASPP branches & project
    # x = Concatenate()([b4, b0, b1, b2, b3])
    # x = Conv2D(256, (1, 1), padding='same',
    #            use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    # x = Activation(activation)(x)
    # x = Dropout(0.5)(x)
    #
    # # DeepLab v.3+ decoder
    #
    # # Feature projection
    # # x4 (x2) block
    #
    #
    # # d
    # x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
    #                                     int(np.ceil(input_shape[1] / 4))))(x) # 128, 256 @256
    # dec_skip1 = Conv2D(48, (1, 1), padding='same',
    #                    use_bias=False, name='feature_projection0')(skip1) # 32, 128, 48
    # dec_skip1 = BatchNormalization(
    #     name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    # dec_skip1 = Activation(activation)(dec_skip1)
    # x = Concatenate()([x, dec_skip1])
    # x = SepConv_BN(x, 256, 'decoder_conv0',
    #                depth_activation=True, epsilon=1e-5)
    # x = Dropout(0.5)(x)
    # x = SepConv_BN(x, 256, 'decoder_conv1',
    #                depth_activation=True, epsilon=1e-5)
    # x = Dropout(0.1)(x)
    #
    # x = Conv2D(classes, (1, 1), padding='same', name='custom_logits_semantic')(x)
    # x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)



    # return input_tensor, x
    # #return img_input, x