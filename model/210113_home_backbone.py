import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
from keras.activations import sigmoid
from keras.regularizers import l2
from keras.layers.experimental.preprocessing import Resizing

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda , Dropout ,BatchNormalization, Multiply, DepthwiseConv2D, SeparableConv2D, Conv2DTranspose
from keras import backend as K




source_layers_to_extract = {
    'B0': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B1': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B2': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B3': ['block3c_add', 'block5e_add', 'block7b_add'],
    'B4': ['block3d_add', 'block5f_add', 'block7b_add'],
    'B5': ['block3e_add', 'block5g_add', 'block7c_add'],
    'B6': ['block3f_add', 'block5h_add', 'block7c_add'],
    'B7': ['block3g_add', 'block5j_add', 'block7d_add'],
}


#    f   k  s    p
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





def create_base_model(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300]):
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



# def convolution(input_tensor, channel, size, stride, padding, name):
#     kernel_size = (size, size)
#     kernel_stride = (stride, stride)
#     conv = Conv2D(channel, kernel_size, kernel_stride, padding=padding, kernel_regularizer=l2(0.0005),
#            kernel_initializer='he_normal', name=name)(input_tensor)
#     conv = BatchNormalization(axis=3, name=name+'_bn')(conv)
#     conv = Activation('relu', name=name+'_relu')(conv)
#
#     return conv



def MBConv(x, expand, squeeze, name):
    r = Conv2D(expand, (1,1), padding='same', name=name+'_expand')(x)
    r = BatchNormalization(axis=3, name=name + '_expand_bn')(r)
    r = Activation(tf.nn.relu6, name=name + '_expand_relu6')(r)

    r = DepthwiseConv2D((3,3), padding='same', name=name+'_depthwise')(r)
    r = BatchNormalization(axis=3, name=name + '_depthwise_bn')(r)
    r = Activation(tf.nn.relu6, name=name + '_depthwise_relu6')(r)

    r = Conv2D(squeeze, (1,1), padding='same', name=name+'_squeeze')(r)
    r = BatchNormalization(axis=3, name=name + '_squeeze_bn')(r)

    return Add()([r, x])

def attentionMBConv(x, expand, squeeze, name):

    r = Conv2D(expand, (1, 1), padding='same', name=name + '_expand')(x)
    r = BatchNormalization(axis=3, name=name + '_expand_bn')(r)
    r = Activation(tf.nn.relu6, name=name + '_expand_relu6')(r)

    r = DepthwiseConv2D((3, 3), padding='same', name=name + '_depthwise')(r)
    r = BatchNormalization(axis=3, name=name + '_depthwise_bn')(r)
    r = Activation(tf.nn.relu6, name=name + '_depthwise_relu6')(r)

    r = Conv2D(squeeze, (1, 1), padding='same', name=name + '_squeeze')(r)
    input_feature = BatchNormalization(axis=3, name=name + '_squeeze_bn')(r)

    ratio = 8

    channel = input_feature.shape[3]
    print('channel=', channel)

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    input_feature = multiply([input_feature, cbam_feature])




    return Add()([input_feature, x])


def strideMBConv(x, expand, squeeze, name):
    r = Conv2D(expand, (1, 1), padding='same', name=name + '_stride_expand')(x)
    r = BatchNormalization(axis=3, name=name + '_stride_expand_bn')(r)
    r = Activation(tf.nn.relu6, name=name + '_stride_expand_relu6')(r)

    r = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', name=name + '_stride_depthwise')(r)
    r = BatchNormalization(axis=3, name=name + '_stride_depthwise_bn')(r)
    r = Activation(tf.nn.relu6, name=name + '_stride_depthwise_relu6')(r)

    r = Conv2D(squeeze, (1,1), padding='same', name=name+'_squeeze')(r)
    r = BatchNormalization(axis=3, name=name + '_squeeze_bn')(r)

    return r

# def deconvolution(input_tensor, channel, size, name):
#     resized = Resizing(size, size, name=name+'_resizing')(input_tensor)
#     return resized

# def deconvolution(input_tensor, channel, size, name):
#     deconv = Conv2DTranspose(channel, (3,3), strides=(2,2),activation='relu', padding='same',name=name+'_deconv'
#                              ,output_padding=(2,2))(input_tensor)
#     print(name+'deconv feature : ', deconv)
#     return deconv
#



def create_backbone(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300], regularization=5e-4):
    source_layers = []
    base = create_base_model(base_model_name, pretrained, IMAGE_SIZE)
    print(base)
    layer_names = source_layers_to_extract[base_model_name]
    print("layer_names : ", layer_names)

    # get extra layer
    #conv75 = base.get_layer('block2b_add').output  # 75 75 24
    conv38 = base.get_layer(layer_names[0]).output # 38 38 40
    #conv19 = base.get_layer(layer_names[1]).output # 19 19 112`
    #conv10 = base.get_layer(layer_names[2]).output # 10 10 320
    print("input")
    print("conv38", conv38)
    #print("conv19", conv19)
    #print("conv10", conv10)


    conv38 = MBConv(conv38, 240, 40, 'conv38_mbconv_1')
    conv38 = MBConv(conv38, 240, 40, 'conv38_mbconv_2')
    conv38 = MBConv(conv38, 240, 40, 'conv38_mbconv_3')
    conv38 = MBConv(conv38, 240, 40, 'conv38_mbconv_4')
    conv38 = MBConv(conv38, 240, 40, 'conv38_mbconv_5')
    conv38 = attentionMBConv(conv38, 240, 40, 'conv38_attentionmbconv_1')

    conv19 = strideMBConv(conv38, 240, 80, 'conv38_stridembconv_1')
    conv19 = MBConv(conv19, 480, 80, 'conv19_mbconv_1')
    conv19 = MBConv(conv19, 480, 80, 'conv19_mbconv_2')
    conv19 = MBConv(conv19, 480, 80, 'conv19_mbconv_3')
    conv19 = MBConv(conv19, 480, 80, 'conv19_mbconv_4')
    conv19 = attentionMBConv(conv19, 480, 80, 'conv19_attentionmbconv_1')

    conv10 = strideMBConv(conv19, 480, 160, 'conv19_stridembconv_1')
    conv10 = MBConv(conv10, 960, 160, 'conv10_mbconv_1')
    conv10 = MBConv(conv10, 960, 160, 'conv10_mbconv_2')
    conv10 = MBConv(conv10, 960, 160, 'conv10_mbconv_3')
    conv10 = MBConv(conv10, 960, 160, 'conv10_mbconv_4')
    conv10 = attentionMBConv(conv10, 960, 160, 'conv10_attentionmbconv_1')

    # fpn conv
    source_layers.append(conv38)
    source_layers.append(conv19)
    source_layers.append(conv10)


    # original code
    # for name in layer_names:
    #     source_layers.append(base.get_layer(name).output)

    x = source_layers[-1]
    # source_layers_0, # block3b_add/add_1:0    38, 38, 40

    # source_layers_1, # block5c_add/add_1:0    19, 19, 112

    # source_layers_2, # block7a_project_bn/cond_1/Identity:0    10, 10, 320

    source_layers.extend(add_extras(extra_layers_params, x))

    return base.input, source_layers