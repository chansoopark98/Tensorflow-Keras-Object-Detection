import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
from keras.activations import sigmoid
from keras.regularizers import l2
from keras.layers.experimental.preprocessing import Resizing

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Concatenate, \
    Conv2D, Add, Activation, Dropout ,BatchNormalization, DepthwiseConv2D
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


# 필터 개수, 커널크기, stride, 패딩
extra_layers_params = [[(128, 1, 1, 'same'), (256, 3, 2, 'same')],
                       [(128, 1, 1, 'same'), (256, 3, 1, 'valid')],
                       [(128, 1, 1, 'same'), (256, 3, 1, 'valid')]]


def remove_dropout(model):
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    del model

    return model_copy





def add_extras(extras, x, regularization=5e-4):
    # 5,5 / 3,3 / 1,1 세 개 수행
    features = []
    for extra_layer in extras:
        x = add_layer(extra_layer[0], x, regularization)
        x = add_layer(extra_layer[1], x, regularization)
        features.append(x)

    return features





def add_layer(layer_params, x, regularization=5e-4):
    filters, kernel_size, stride, padding = layer_params
    x = Conv2D(filters, kernel_size, stride, padding=padding, kernel_regularizer=l2(regularization))(x)
    x = Activation('relu')(x)

    return x





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


def convolution(input_tensor, channel, size, stride, padding, name):
    kernel_size = (size, size)
    kernel_stride = (stride, stride)
    conv = Conv2D(channel, kernel_size, kernel_stride, padding=padding, kernel_regularizer=l2(0.0005),
           kernel_initializer='he_normal', name=name)(input_tensor)
    conv = BatchNormalization(axis=3, name=name+'_bn')(conv)
    conv = Activation('relu', name=name+'_relu')(conv)

    return conv

def CA(x, expand, name):
    squeeze = x.shape[3]

    r = Conv2D(expand, (1, 1), padding='same', kernel_regularizer=l2(0.0005),
           kernel_initializer='he_normal', name=name + '_stride_expand')(x)
    r = BatchNormalization(axis=3, name=name + '_expand_bn')(r)
    r = Activation('relu', name=name + '_relu')(r)

    r = DepthwiseConv2D((3,3), padding='same', depthwise_regularizer=l2(0.0005),
                        depthwise_initializer='he_normal', name=name+'_depthwise')(r)
    r = BatchNormalization(axis=3, name=name + '_depthwise_bn')(r)
    r = Activation('relu', name=name + '_depthwise_relu')(r)

    shared_layer_one = Dense(expand // 8,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(expand,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(r)
    avg_pool = Reshape((1, 1, expand))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)


    max_pool = GlobalMaxPooling2D()(r)
    max_pool = Reshape((1, 1, expand))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)

    r = multiply([r, channel_attention])

    r = Conv2D(squeeze, (1, 1), padding='same', kernel_regularizer=l2(0.0005),
           kernel_initializer='he_normal', name=name + '_squeeze_conv')(r)
    r = BatchNormalization(axis=3, name=name + '_squeeze_bn')(r)

    return r


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
    #conv75 = base.get_layer('block2b_add').output  # 75 75 24
    efficient_conv38 = base.get_layer(layer_names[0]).output # 38 38 40
    efficient_conv19 = base.get_layer(layer_names[1]).output # 19 19 112`
    efficient_conv10 = base.get_layer(layer_names[2]).output # 10 10 320
    print("input")
    print("conv38", efficient_conv38)
    print("conv19", efficient_conv19)
    print("conv10", efficient_conv10)

    conv38 = convolution(efficient_conv38, 64, 3, 1, 'SAME', 'conv38_resampling')
    conv19 = convolution(efficient_conv19, 128, 3, 1, 'SAME', 'conv19_resampling')
    conv10 = convolution(efficient_conv10, 256, 3, 1, 'SAME', 'conv10_resampling')

    conv38 = CA(conv38, 128, 'conv38_ca')
    conv19 = CA(conv19, 256, 'conv19_ca')
    conv10 = CA(conv10, 512, 'conv10_ca')

    conv10_upSampling = upSampling(conv10, 19, 'conv10_to_conv19') # 10x10@256 to 19x19@256
    conv10_upSampling = convolution(conv10_upSampling, 128, 1, 1, 'SAME', 'conv10_upSampling_conv')
    concat_conv19 = Concatenate()([conv10_upSampling, conv19]) # 19x19 / @128+128


    conv19_upSampling = upSampling(conv19, 38, 'conv19_to_conv38')  # 19x19@128 to 38x38@128
    conv19_upSampling = convolution(conv19_upSampling, 64, 1, 1, 'SAME', 'conv19_upSampling_conv')
    concat_conv38 = Concatenate()([conv19_upSampling, conv38]) # 38x38 / @64+64


    conv38 = convolution(concat_conv38, 128, 3, 1, 'SAME', 'conv38_for_predict')
    conv19 = convolution(concat_conv19, 256, 3, 1, 'SAME', 'conv19_for_predict')
    conv10 = convolution(conv10, 256, 3, 1, 'SAME', 'conv10_for_predict')

    # fpn conv
    source_layers.append(conv38)
    source_layers.append(conv19)
    source_layers.append(conv10)


    # original
    # for name in layer_names:
    #     source_layers.append(base.get_layer(name).output)
    x = source_layers[-1]
    # source_layers_0, # block3b_add/add_1:0    38, 38, 40

    # source_layers_1, # block5c_add/add_1:0    19, 19, 112

    # source_layers_2, # block7a_project_bn/cond_1/Identity:0    10, 10, 320

    source_layers.extend(add_extras(extra_layers_params, x))

    return base.input, source_layers