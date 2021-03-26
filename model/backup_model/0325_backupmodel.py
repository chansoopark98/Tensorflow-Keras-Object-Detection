import efficientnet.keras as efn
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D,  Reshape, Dense, multiply, Concatenate, \
    Conv2D, Add, Activation, Dropout ,BatchNormalization, DepthwiseConv2D, Lambda ,  UpSampling2D, SeparableConv2D, MaxPooling2D
from tensorflow.keras import backend as K
from functools import reduce

activation = tf.keras.activations.swish
# activation = tfa.activations.mish

MOMENTUM = 0.997
EPSILON = 1e-4

GET_EFFICIENT_NAME = {
    'B0': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B1': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B2': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B3': ['block3c_add', 'block5e_add', 'block7b_add'],
    'B4': ['block3d_add', 'block5f_add', 'block7b_add'],
    'B5': ['block3e_add', 'block5g_add', 'block7c_add'],
    'B6': ['block3f_add', 'block5h_add', 'block7c_add'],
    'B7': ['block3g_add', 'block5j_add', 'block7d_add'],
}

CONV_KERNEL_INITALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def remove_dropout(model):
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    del model

    return model_copy



def create_efficientNet(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512]):
    if pretrained is False:
        weights = None

    else:
        weights = "imagenet"

    if base_model_name == 'B0':
        base = efn.EfficientNetB0(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B1':
        base = efn.EfficientNetB1(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B2':
        base = efn.EfficientNetB2(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B3':
        base = efn.EfficientNetB3(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B4':
        base = efn.EfficientNetB4(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B5':
        base = efn.EfficientNetB5(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B6':
        base = efn.EfficientNetB6(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B7':
        base = efn.EfficientNetB7(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    base = remove_dropout(base)
    base.trainable = True

    return base

def MBConv(input_tensor, stride, name):
    expansion = 3
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(input_tensor)[channel_axis]

    r = Conv2D(expansion * in_channels, (1, 1), kernel_regularizer=l2(0.0005), kernel_initializer=CONV_KERNEL_INITALIZER,
               padding='same', name=name + '_mbconv_expansion_conv')(input_tensor)
    r = BatchNormalization(axis=channel_axis, name = name + '_mbconv_expansion_bn')(r)
    r = Activation(activation, name=name + '_mbconv_expansion_relu')(r)

    # r = DepthwiseConv2D((3, 3), strides=stride, depthwise_regularizer=l2(0.0005), depthwise_initializer='he_normal',
    #                     activation=None, use_bias=False,
    #                     padding='same', name=name + '_mbconv_squeeze_depthwise')(r)
    # Default
    r = DepthwiseConv2D((3, 3), strides=stride, activation=None, use_bias=False, kernel_initializer=CONV_KERNEL_INITALIZER,
                        padding='same', name=name + '_mbconv_squeeze_depthwise')(r)
    r = BatchNormalization(axis=channel_axis, name=name + '_mbconv_squeeze_depthwise_bn')(r)
    r = Activation(activation, name=name + '_mbconv_squeeze_depthwise_relu')(r)

    shared_layer_one = Dense(expansion * in_channels // 16,
                             activation=activation,
                             kernel_initializer=DENSE_KERNEL_INITIALIZER,
                             use_bias=True
                             )
    shared_layer_two = Dense(expansion * in_channels,
                             kernel_initializer=DENSE_KERNEL_INITIALIZER,
                             use_bias=True
                             )

    avg_pool = GlobalAveragePooling2D()(r)
    avg_pool = Reshape((1, 1, expansion * in_channels))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    channel_attention = Activation('sigmoid')(avg_pool)

    r = multiply([r, channel_attention])



    r = Conv2D(in_channels, (1, 1), kernel_regularizer=l2(0.0005), kernel_initializer=CONV_KERNEL_INITALIZER,
               padding='same', name=name + '_mbconv_squeeze_conv')(r)
    r = BatchNormalization(axis=channel_axis, name=name + '_mbconv_squeeze_bn')(r)

    if stride == 2:
        return r
    return Add(name=name+'residual_add')([input_tensor, r])

def extraMBConv(x, padding, name, stride=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]

    r = Conv2D(in_channels, (1, 1), kernel_regularizer=l2(0.0005), kernel_initializer='he_normal',
               padding='same', name=name + '_mbconv_squeeze_1')(x)
    r = BatchNormalization(axis=channel_axis, name=name + '_mbconv_squeeze_bn_1')(r)
    r = Activation(activation, name=name + '_mbconv_squeeze_relu_1')(r)

    r = DepthwiseConv2D((3, 3), strides=stride, padding=padding, activation=None, use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITALIZER,
                        name=name + '_mbconv_squeeze_depthwise')(r)
    r = BatchNormalization(axis=channel_axis, name=name + '_mbconv_squeeze_depthwise_bn')(r)
    r = Activation(activation, name=name + '_mbconv_squeeze_depthwise_relu')(r)

    r = Conv2D(in_channels, (1, 1), kernel_regularizer=l2(0.0005), kernel_initializer='he_normal',
               padding='same', name=name + '_mbconv_squeeze_conv')(r)
    r = BatchNormalization(axis=channel_axis, name=name + '_mbconv_squeeze_bn')(r)
    return r

def convolution(input_tensor, channel, size, stride, padding, name):
    kernel_size = (size, size)
    kernel_stride = (stride, stride)
    conv = Conv2D(channel, kernel_size, kernel_stride, padding=padding, kernel_regularizer=l2(0.0005),
           kernel_initializer=CONV_KERNEL_INITALIZER, name=name)(input_tensor)
    conv = BatchNormalization(axis=3, name=name+'_bn')(conv)
    conv = Activation(activation, name=name+'_relu')(conv)

    return conv




def SA(x):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]

    dilated_feature = DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=False, dilation_rate=(3, 3), padding='same')(x)
    depthwise_feature = DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=False, padding='same')(x)

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(dilated_feature)

    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(depthwise_feature)

    concat = Concatenate(axis=3)([avg_pool, max_pool])

    sa_feature = Conv2D(filters=1,
                          kernel_size=7,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer=CONV_KERNEL_INITALIZER,
                          use_bias=False)(concat)

    return multiply([x, sa_feature])

num_channels = [64, 88, 112, 160, 224, 288, 384]

def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=f'{name}/conv')
    f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

def build_BiFPN(p3, p5, p7, num_channels=64 , id=0, freeze_bn=False):
    if id == 0:
        C3, C4, C5 = p3, p5 ,p7
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
        P6_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = UpSampling2D()(P7_in)
        P6_td = Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = UpSampling2D()(P6_td)
        P5_td = Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = UpSampling2D()(P5_td)
        P4_td = Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P3_in = Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = UpSampling2D()(P4_td)
        P3_out = Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

        print('p3', P3_out)
        print('p4', P4_td)
        print('p5', P5_td)
        print('p6', P6_td)
        print('p7', P7_out)
        return P3_out, P4_td, P5_td, P6_td, P7_out

    # else:
    #     P3_in, P4_in, P5_in, P6_in, P7_in = features
    #     P7_U = layers.UpSampling2D()(P7_in)
    #     P6_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
    #     P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
    #     P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
    #                                name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
    #     P6_U = layers.UpSampling2D()(P6_td)
    #     P5_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
    #     P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
    #     P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
    #                                name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
    #     P5_U = layers.UpSampling2D()(P5_td)
    #     P4_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
    #     P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
    #     P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
    #                                name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
    #     P4_U = layers.UpSampling2D()(P4_td)
    #     P3_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
    #     P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
    #     P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
    #                                 name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
    #     P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
    #     P4_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
    #     P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
    #     P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
    #                                 name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)
    #
    #     P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
    #     P5_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
    #     P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
    #     P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
    #                                 name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)
    #
    #     P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
    #     P6_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
    #     P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
    #     P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
    #                                 name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)
    #
    #     P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
    #     P7_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
    #     P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
    #     P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
    #                                 name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)






def upSampling(input_tensor,name):
    # resized = Resizing(size, size, name=name+'_resizing')(input_tensor)
    resized = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_tensor)
    return resized

def csnet_extra_model(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], regularization=5e-4):
    source_layers = []
    base = create_efficientNet(base_model_name, pretrained, IMAGE_SIZE)

    layer_names = GET_EFFICIENT_NAME[base_model_name]

    # get extra layer
    #efficient_conv75 = base.get_layer('block2b_add').output  # 75 75 24
    conv38 = base.get_layer(layer_names[0]).output # 64 64 40
    conv19 = base.get_layer(layer_names[1]).output # 32 32 112
    conv10 = base.get_layer(layer_names[2]).output # 16 16 320
    build_BiFPN(conv38, conv19, conv10, 64, 0)

    # bottom-up pathway
    conv10_upSampling = upSampling(conv10, 'conv10_to_conv19')  # 10x10@256 to 19x19@256

    sa_conv10 = SA(conv10)

    concat_conv19 = Concatenate()([conv10_upSampling, conv19])
    concat_conv19_1x1 = convolution(concat_conv19, 256, 1, 1, 'same', 'concat_conv19_1x1_channel')

    sa_conv19 = SA(concat_conv19_1x1)

    concat_conv19_1x1 = MBConv(concat_conv19_1x1, 1, 'conv19_upSampling_conv') # for top-down
    #ca_conv19 = CA(concat_conv19)
    ca_conv19 = upSampling(concat_conv19_1x1, 'conv19_to_conv38')  # 10x10@128 to 19x19@128

    concat_conv38 = Concatenate()([conv38, ca_conv19])  # 38x39 / @64+128
    concat_conv38 = convolution(concat_conv38, 128, 1, 1, 'same', 'concat_conv38_1x1_channel')
    concat_conv38 = MBConv(concat_conv38, 1, 'conv38_upSampling_conv')

    sa_conv38 = SA(concat_conv38)

    # top-down pathway
    down_conv19 = MBConv(sa_conv38, 2, 'conv38_downSampling_conv') # STRIDE = 2
    down_concat_conv19 = Concatenate()([sa_conv19, down_conv19]) # 19x19@ 64 + 128
    down_concat_conv19 = convolution(down_concat_conv19, 256, 1, 1, 'same', 'concat_conv19_1x1_channel_2')
    down_conv10 = MBConv(down_concat_conv19, 2,  'conv10_downSampling_conv')

    down_concat_conv10 = Concatenate()([sa_conv10, down_conv10])  # @256+128
    down_concat_conv10 = convolution(down_concat_conv10, 256, 1, 1, 'same', 'concat_conv10_1x1_channel_2')

    conv5 = extraMBConv(down_concat_conv10, 'same', 'conv10_to_conv5_1', (1, 1))
    conv5 = extraMBConv(conv5, 'same', 'conv10_to_conv5_2', (2, 2))

    conv3 = extraMBConv(conv5, 'same','conv5_to_conv3_1',(1, 1))
    conv3 = extraMBConv(conv3, 'same', 'conv5_to_conv3_2',(2, 2))


    conv1 = extraMBConv(conv3, 'same', 'conv3_to_conv1_1')
    conv1 = extraMBConv(conv1, 'valid', 'conv3_to_conv1_2')

    conv0 = extraMBConv(conv1, 'same', 'conv1_to_conv0_1')
    conv0 = extraMBConv(conv0, 'same', 'conv0_1_to_conv0_2', (2, 2))

    # predict features
    source_layers.append(sa_conv38)
    source_layers.append(down_concat_conv19)
    source_layers.append(down_concat_conv10)
    source_layers.append(conv5)
    source_layers.append(conv3)
    source_layers.append(conv1)
    source_layers.append(conv0)
    print(concat_conv38)
    print(down_concat_conv19)
    print(down_concat_conv10)
    print(conv5)
    print(conv3)
    print(conv1)
    print(conv0)

    return base.input, source_layers