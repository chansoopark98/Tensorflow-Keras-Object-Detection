import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
from keras.activations import sigmoid
from keras.regularizers import l2
from keras.layers.experimental.preprocessing import Resizing


from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Concatenate, \
    Conv2D, Add, Activation, Dropout ,BatchNormalization, DepthwiseConv2D, Lambda , MaxPool2D
from keras import backend as K


get_efficient_feature = {
    'B0': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B1': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B2': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B3': ['block3c_add', 'block5e_add', 'block7b_add'],
    'B4': ['block3d_add', 'block5f_add', 'block7b_add'],
    'B5': ['block3e_add', 'block5g_add', 'block7c_add'],
    'B6': ['block3f_add', 'block5h_add', 'block7c_add'],
    'B7': ['block3g_add', 'block5j_add', 'block7d_add'],
}



def remove_dropout(model):
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    del model

    return model_copy



def create_efficientNet(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300]):
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

    r = Conv2D(expansion * in_channels, (1, 1), kernel_regularizer=l2(0.0005), kernel_initializer='he_normal',
               padding='same', name=name + '_mbconv_expansion_conv')(input_tensor)
    r = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999,
                           name = name + '_mbconv_expansion_bn')(r)
    r = Activation('relu', name=name + '_mbconv_expansion_relu')(r)

    # r = DepthwiseConv2D((3, 3), strides=stride, depthwise_regularizer=l2(0.0005), depthwise_initializer='he_normal',
    #                     activation=None, use_bias=False,
    #                     padding='same', name=name + '_mbconv_squeeze_depthwise')(r)
    # Default
    r = DepthwiseConv2D((3, 3), strides=stride, activation=None, use_bias=False,
                        padding='same', name=name + '_mbconv_squeeze_depthwise')(r)
    r = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999,
                           name=name + '_mbconv_squeeze_depthwise_bn')(r)
    r = Activation('relu', name=name + '_mbconv_squeeze_depthwise_relu')(r)

    shared_layer_one = Dense(expansion * in_channels // 16,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(expansion * in_channels,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(r)
    avg_pool = Reshape((1, 1, expansion * in_channels))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    channel_attention = Activation('sigmoid')(avg_pool)

    r = multiply([r, channel_attention])



    r = Conv2D(in_channels, (1, 1), kernel_regularizer=l2(0.0005), kernel_initializer='he_normal',
               padding='same', name=name + '_mbconv_squeeze_conv')(r)
    r = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999,
                           name=name + '_mbconv_squeeze_bn')(r)

    if stride == 2:
        return r
    return Add(name=name+'residual_add')([input_tensor, r])

def extraMBConv(x, padding, name, stride=(1, 1)):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]

    r = Conv2D(in_channels, (1, 1), kernel_regularizer=l2(0.0005), kernel_initializer='he_normal',
               padding='same', name=name + '_mbconv_squeeze_1')(x)
    r = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999,
                           name=name + '_mbconv_squeeze_bn_1')(r)
    r = Activation('relu', name=name + '_mbconv_squeeze_relu_1')(r)

    r = DepthwiseConv2D((3, 3), strides=stride ,padding=padding, name=name + '_mbconv_squeeze_depthwise')(r)
    r = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999,
                           name=name + '_mbconv_squeeze_depthwise_bn')(r)
    r = Activation('relu', name=name + '_mbconv_squeeze_depthwise_relu')(r)

    r = Conv2D(in_channels, (1, 1), kernel_regularizer=l2(0.0005), kernel_initializer='he_normal',
               padding='same', name=name + '_mbconv_squeeze_conv')(r)
    r = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999,
                           name=name + '_mbconv_squeeze_bn')(r)
    return r



def convolution(input_tensor, channel, size, stride, padding, name):
    kernel_size = (size, size)
    kernel_stride = (stride, stride)
    conv = Conv2D(channel, kernel_size, kernel_stride, padding=padding, kernel_regularizer=l2(0.0005),
           kernel_initializer='he_normal', name=name)(input_tensor)
    conv = BatchNormalization(axis=3, name=name+'_bn')(conv)
    conv = Activation('relu', name=name+'_relu')(conv)

    return conv

def dilated_convolution(input_tensor, channel, size, stride, dilated_rate, padding, name):
    kernel_size = (size, size)
    kernel_stride = (stride, stride)
    dilated_size = (dilated_rate, dilated_rate)
    conv = Conv2D(channel, kernel_size, kernel_stride, dilation_rate=dilated_size,
                  padding=padding, kernel_regularizer=l2(0.0005),
                  kernel_initializer='he_normal', name=name)(input_tensor)
    conv = BatchNormalization(axis=3, name=name+'_bn')(conv)
    conv = Activation('relu', name=name+'_relu')(conv)

    return conv


def CA(x):
    channel = x.shape[3]

    shared_layer_one = Dense(channel // 16,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)


    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)

    r = multiply([x, channel_attention])
    return r

def SA(x):
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)

    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)

    concat = Concatenate(axis=3)([avg_pool, max_pool])

    sa_feature = Conv2D(filters=1,
                          kernel_size=7,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([x, sa_feature])



def upSampling(input_tensor, size, name):
    resized = Resizing(size, size, name=name+'_resizing')(input_tensor)
    return resized



def csnet_extra_model(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300], regularization=5e-4):
    source_layers = []
    base = create_efficientNet(base_model_name, pretrained, IMAGE_SIZE)
    print(base)
    layer_names = get_efficient_feature[base_model_name]
    print("layer_names : ", layer_names)

    # get extra layer
    #efficient_conv75 = base.get_layer('block2b_add').output  # 75 75 24
    efficient_conv38 = base.get_layer(layer_names[0]).output # 38 38 40
    efficient_conv19 = base.get_layer(layer_names[1]).output # 19 19 112`
    efficient_conv10 = base.get_layer(layer_names[2]).output # 10 10 320
    print("input")
    #print("conv75", efficient_conv75)
    print("conv38", efficient_conv38)
    print("conv19", efficient_conv19)
    print("conv10", efficient_conv10)



    #conv38 = convolution(efficient_conv38, 64, 3, 1, 'same', 'conv38_channel_64')
    conv38 = MBConv(efficient_conv38, 1, 'conv38_channel_64')
    #conv38 = CA(conv38)
    #conv38 = SA(conv38)

    #conv19 = convolution(efficient_conv19, 128, 3, 1, 'same', 'conv19_channel_128')
    conv19 = MBConv(efficient_conv19, 1, 'conv19_channel_128')
    #conv19 = CA(conv19)
    #conv19 = SA(conv19)

    #conv10 = convolution(efficient_conv10, 256, 3, 1, 'same', 'conv10_channel_256')
    conv10 = MBConv(efficient_conv10, 1, 'conv10_channel_256')
    #conv10 = CA(conv10)
    #conv10 = SA(conv10)

    # bottom-up pathway


    conv10_upSampling = upSampling(conv10, 19, 'conv10_to_conv19')  # 10x10@256 to 19x19@256

    concat_conv19 = Concatenate()([conv10_upSampling, conv19])
    concat_conv19 = convolution(concat_conv19, 128, 1, 1, 'same', 'concat_conv19_1x1_channel')
    concat_conv19 = MBConv(concat_conv19, 1, 'conv19_upSampling_conv') # for top-down
    #ca_conv19 = CA(concat_conv19)
    ca_conv19 = upSampling(concat_conv19, 38, 'conv19_to_conv38')  # 10x10@128 to 19x19@128

    concat_conv38 = Concatenate()([conv38, ca_conv19])  # 38x39 / @64+128
    concat_conv38 = convolution(concat_conv38, 64, 1, 1, 'same', 'concat_conv38_1x1_channel')
    concat_conv38 = MBConv(concat_conv38, 1, 'conv38_upSampling_conv')
    # sa_conv38 = SA(concat_conv38)


    # Mid-Bridge pathway
    bridge_conv38 = MBConv(concat_conv38, 1, 'conv38_bridge_1')
    bridge_conv38 = MBConv(bridge_conv38, 1, 'conv38_bridge_2') # for predict --------
    #bridge_conv38 = CA(bridge_conv38)
    bridge_conv38 = SA(bridge_conv38)
    print(' bridge_conv38 -- ' , bridge_conv38)

    bridge_conv19 = MBConv(concat_conv19, 1, 'conv19_bridge_1')
    bridge_conv19 = MBConv(bridge_conv19, 1, 'conv19_bridge_2')
    #bridge_conv19 = CA(bridge_conv19)
    bridge_conv19 = SA(bridge_conv19)
    print(' bridge_conv39  --  ', bridge_conv19)

    bridge_conv10 = MBConv(conv10, 1, 'conv10_bridge_1')
    bridge_conv10 = MBConv(bridge_conv10, 1, 'conv10_bridge_2')
    #bridge_conv10 = CA(bridge_conv10)
    bridge_conv10 = SA(bridge_conv10)
    print(' bridge_conv10  --  ', bridge_conv10)


    # top-down pathway
    down_conv19 = MBConv(bridge_conv38, 2, 'conv38_downSampling_conv') # STRIDE = 2
    down_concat_conv19 = Concatenate()([bridge_conv19, down_conv19]) # 19x19@ 64 + 128
    down_concat_conv19 = convolution(down_concat_conv19, 128, 1, 1, 'same', 'concat_conv19_1x1_channel_2')
    down_concat_conv19 = MBConv(down_concat_conv19, 1, 'conv19_down_conv') # for predict --------
    #sa_down_conv19 = SA(down_concat_conv19) ### for predict

    down_conv10 = MBConv(down_concat_conv19, 2,  'conv10_downSampling_conv')
    down_concat_conv10 = Concatenate()([bridge_conv10, down_conv10])  # @256+128
    down_concat_conv10 = convolution(down_concat_conv10, 256, 1, 1, 'same', 'concat_conv10_1x1_channel_2')
    down_concat_conv10 = MBConv(down_concat_conv10, 1, 'conv10_down_conv') # for predict -------
    #sa_conv_conv10 = SA(down_concat_conv10) ### for predict


    #sa_conv38 = MBConv(sa_conv38, 1,  'conv38_for_predict')
    #sa_down_conv19 = MBConv(sa_down_conv19, 1, 'conv19_for_predict')
    #sa_conv_conv10 = MBConv(sa_conv_conv10, 1, 'conv10_for_predict')


    conv5 = extraMBConv(down_concat_conv10, 'same', 'conv10_to_conv5_1', (1, 1))
    conv5 = extraMBConv(conv5, 'same', 'conv10_to_conv5_2', (2, 2))
    #conv5 = CA(conv5)
    #conv5 = SA(conv5)

    conv3 = extraMBConv(conv5, 'same','conv5_to_conv3_1')
    conv3 = extraMBConv(conv3, 'valid', 'conv5_to_conv3_2')
    #conv3 = CA(conv3)
    #conv3 = SA(conv3)

    conv1 = extraMBConv(conv3, 'same', 'conv3_to_conv1_1')
    conv1 = extraMBConv(conv1, 'valid', 'conv3_to_conv1_2')
    #conv1 = CA(conv1)
    #conv1 = SA(conv1)

    # predict features
    source_layers.append(bridge_conv38)
    source_layers.append(down_concat_conv19)
    source_layers.append(down_concat_conv10)
    source_layers.append(conv5)
    source_layers.append(conv3)
    source_layers.append(conv1)


    return base.input, source_layers