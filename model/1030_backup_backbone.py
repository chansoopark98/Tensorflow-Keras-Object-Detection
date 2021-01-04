import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
from keras.activations import sigmoid
from keras.regularizers import l2
from keras.layers.experimental.preprocessing import Resizing

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda , Dropout ,BatchNormalization, Multiply
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



def convolution(input_tensor, channel, size, stride, padding, name):
    kernel_size = (size, size)
    kernel_stride = (stride, stride)
    conv = Conv2D(channel, kernel_size, kernel_stride, padding=padding, kernel_regularizer=l2(0.0005),
           kernel_initializer='he_normal', name=name)(input_tensor)
    conv = BatchNormalization(axis=-1, name=name+'_bn')(conv)
    conv = Activation('relu', name=name+'_relu')(conv)

    return conv

# shared dense layer

def concat_convolution(input_tensor, channel, size, stride, padding, name):
    kernel_size = (size, size)
    kernel_stride = (stride, stride)
    refine_feature = Conv2D(channel, kernel_size, kernel_stride, padding=padding, kernel_regularizer=l2(0.0005),
           kernel_initializer='he_normal', name=name)(input_tensor)
    refine_feature = BatchNormalization(axis=-1, name=name+'_bn')(refine_feature)
    refine_feature = Activation('relu', name=name+'_relu')(refine_feature)

    # Residual block
    if channel <= 128 :
        residual = Conv2D(channel, (3, 3), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
               kernel_initializer='he_normal', name=name+'residual_1')(refine_feature)
        residual = Activation('relu', name=name+'residual_1_relu')(residual)
        residual = Conv2D(channel, (3, 3), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
                      kernel_initializer='he_normal', name=name + 'residual_1')(residual)

        residual = Add()([refine_feature, residual])
        residual = Activation('relu', name=name + 'residual_final_relu')(residual)

    else :
        residual = Conv2D(channel/4, (1, 1), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
               kernel_initializer='he_normal', name=name+'residual_1')(refine_feature)
        residual = Activation('relu', name=name+'residual_1_relu')(residual)

        residual = Conv2D(channel/4, (3, 3), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
                      kernel_initializer='he_normal', name=name + 'residual_2')(residual)
        residual = Activation('relu', name=name + 'residual_2_relu')(residual)

        residual = Conv2D(channel, (1, 1), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
                          kernel_initializer='he_normal', name=name + 'residual_3')(residual)

        residual = Add()([refine_feature, residual])
        residual = Activation('relu', name=name + 'residual_final_relu')(residual)


    return residual

def attetnion_convolution(input_tensor, channel, size, stride, padding, name):
    kernel_size = (size, size)
    kernel_stride = (stride, stride)
    refine_feature = Conv2D(channel, kernel_size, kernel_stride, padding=padding, kernel_regularizer=l2(0.0005),
           kernel_initializer='he_normal', name=name)(input_tensor)
    refine_feature = BatchNormalization(axis=-1, name=name+'_bn')(refine_feature)
    refine_feature = Activation('relu', name=name+'_relu')(refine_feature)

    # Residual block
    if channel <= 128 :
        residual = Conv2D(channel, (3, 3), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
               kernel_initializer='he_normal', name=name+'residual_1')(refine_feature)
        residual = Activation('relu', name=name+'residual_1_relu')(residual)
        residual = Conv2D(channel, (3, 3), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
                      kernel_initializer='he_normal', name=name + 'residual_1')(residual)

        residual = Add()([refine_feature, residual])
        residual = Activation('relu', name=name + 'residual_final_relu')(residual)

    else :
        residual = Conv2D(channel/4, (1, 1), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
               kernel_initializer='he_normal', name=name+'residual_1')(refine_feature)
        residual = Activation('relu', name=name+'residual_1_relu')(residual)

        residual = Conv2D(channel/4, (3, 3), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
                      kernel_initializer='he_normal', name=name + 'residual_2')(residual)
        residual = Activation('relu', name=name + 'residual_2_relu')(residual)

        residual = Conv2D(channel, (1, 1), (1, 1), padding=padding, kernel_regularizer=l2(0.0005),
                          kernel_initializer='he_normal', name=name + 'residual_3')(residual)

        residual = Add()([refine_feature, residual])
        residual = Activation('relu', name=name + 'residual_final_relu')(residual)

    # feature attetion
    shared_dense = Dense
    emphasize = tf.keras.layers.GlobalAveragePooling2D()(residual)
    emphasize = shared_dense(units=channel/4, activation='relu', name=name+'_shared_dense_1')(emphasize)
    emphasize = shared_dense(units=channel, activation=None, name=name+'_shared_dense_2')(emphasize)
    emphasize = sigmoid(emphasize)
    emphasize = Multiply()([residual, emphasize])
    emphasize = Add()([refine_feature, emphasize])

    return emphasize



def deconvolution(input_tensor, channel, size, name):
    resized = Resizing(size, size, name=name+'_resizing')(input_tensor)
    conv = Conv2D(channel, (3, 3), (1, 1), padding='SAME', kernel_regularizer=l2(0.0005),
           kernel_initializer='he_normal', name=name+'_resize_conv')(resized)
    conv = BatchNormalization(axis=-1, name=name+'_bn')(conv)
    conv = Activation('relu', name=name+'resize_relu')(conv)

    return conv









def create_backbone(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300], regularization=5e-4):
    source_layers = []
    base = create_base_model(base_model_name, pretrained, IMAGE_SIZE)
    print(base)
    layer_names = source_layers_to_extract[base_model_name]
    print("layer_names : ", layer_names)

    # get extra layer
    #conv75 = base.get_layer('block2b_add').output  # 75 75 24
    conv150 = base.get_layer('block1a_activation').output  # 75 75 24
    print("conv150",conv150)
    conv38 = base.get_layer(layer_names[0]).output # 38 38 40
    conv19 = base.get_layer(layer_names[1]).output # 19 19 112`
    conv10 = base.get_layer(layer_names[2]).output # 10 10 320
    print("input")
    print("conv38", conv38)
    print("conv19", conv19)
    print("conv10", conv10)

    conv38 = convolution(conv38, 64, 3, 1, 'SAME', 'conv38_resampling')
    conv19 = convolution(conv19, 128, 3, 1, 'SAME', 'conv19_resampling')
    conv10 = convolution(conv10, 256, 3, 1, 'SAME', 'conv10_resampling')

    # fpn conv

    conv38_concat = convolution(conv38, 128, 3, 2, 'SAME', 'conv38_down_1')
    print("conv38_concat", conv38_concat)

    conv19 = Concatenate()([conv38_concat, conv19]) # 256 = 128 + 128
    conv19 = concat_convolution(conv19, 256, 3, 1, 'SAME', 'conv19_concat_conv')
    print("conv19", conv19)

    conv19_concat = convolution(conv19, 256, 3, 2, 'SAME', 'conv19_down_1')
    #conv19_to_deconv = convolution(conv19, 176, 3, 2, 'VALID', 'for_deconv_concat')
    print("conv19_concat", conv19_concat)
    conv10 = Concatenate()([conv19_concat, conv10])
    # conv10_1 = 예측전용 conv10
    conv10 = concat_convolution(conv10, 512, 3, 1, 'SAME', 'conv10_concat_conv')
    print("CONVCAT_conv10", conv10) # 10x10
    # conv10_@ = deconv용 conv10

    # fpn deconv

    # DECONV TEST
    # 3X3 Deconv
    # same = 20
    # valid = 21
    deconv_conv19 = deconvolution(conv10, 256, 19, 'deconv10_to_19')
    print("deconv_conv19_ conv10_2", deconv_conv19)

    conv19 = Concatenate()([deconv_conv19, conv19])
    conv19 = attetnion_convolution(conv19, 512, 3, 1, 'SAME', 'conv19_concat_deconv')
    print("conv19", conv19)


    #deconv_conv38 = deconvolution(conv19,256,3, 'SAME', 'deconv19_to_38')
    deconv_conv38 = deconvolution(conv19, 256, 38, 'deconv19_to_38')
    print("deconv_conv38", deconv_conv38)
    conv38 = Concatenate()([deconv_conv38, conv38])
    conv38 = attetnion_convolution(conv38, 320, 3, 1, 'SAME', 'conv38_concat_deconv')
    print("conv38", conv38)


    conv38 = convolution(conv38, 320, 3, 1, 'SAME', 'conv38_fpn')
    conv19 = convolution(conv19, 512, 3, 1, 'SAME', 'conv19_fpn')
    conv10 = convolution(conv10, 512, 3, 1, 'SAME', 'conv10_fpn')



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
