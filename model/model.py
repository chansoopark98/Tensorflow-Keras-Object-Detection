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

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]

class weightAdd(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(weightAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(weightAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


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

    dilated_feature = DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=False, dilation_rate=(3, 3), padding='same')(x)
    depthwise_feature = DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=False, padding='same')(x)

    dil_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(dilated_feature)

    dwc_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(depthwise_feature)

    concat = Concatenate(axis=3)([dil_pool, dwc_pool])

    sa_feature = Conv2D(filters=1,
                          kernel_size=7,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          use_bias=False)(concat)

    return multiply([x, sa_feature])



def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=name+'/conv')
    f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))



def build_FPN(features, num_channels=64, times=0, normal_fusion=True, freeze_bn=False):
    if times == 0 :
        C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        P6_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)


        # # Add spatial attention
        # P6_in_2 = SA(P6_in)

        P7_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = UpSampling2D()(P7_in)

        if normal_fusion:
            P6_td = Add(name='fpn_cells/cell_/fnode0/add')([P6_in, P7_U])
        else :
            P6_td = weightAdd(name='fpn_cells/cell_/fnode0/add')([P6_in, P7_U])

        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name='fpn_cells/cell_/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = UpSampling2D()(P6_td)


        if normal_fusion:
            P5_td = Add(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U])
        else :
            P5_td = weightAdd(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U])


        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name='fpn_cells/cell_/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = UpSampling2D()(P5_td)


        if normal_fusion:
            P4_td = Add(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U])
        else :
            P4_td = weightAdd(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U])


        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode2/op_after_combine7')(P4_td)
        P3_in = Conv2D(num_channels, kernel_size=1, padding='same',
                              name='fpn_cells/cell_/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = UpSampling2D()(P4_td)

        if normal_fusion:
            P3_out = Add(name='fpn_cells/cell_/fnode3/add')([P3_in, P4_U])
        else :
            P3_out = weightAdd(name='fpn_cells/cell_/fnode3/add')([P3_in, P4_U])

        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name='fpn_cells/cell_/fnode4/resample_0_1_9/bn')(P4_in_2)

        # # Add spatial attention
        # P4_in_2 = SA(P4_in_2)


        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)

        if normal_fusion:
            P4_out = Add(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td, P3_D])
        else:
            P4_out = weightAdd(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td, P3_D])

        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name='fpn_cells/cell_/fnode5/resample_0_2_10/bn')(P5_in_2)

        # # Add spatial attention
        # P5_in_2 = SA(P5_in_2)


        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

        if normal_fusion:
            P5_out = Add(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D])
        else:
            P5_out = weightAdd(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D])

        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode5/op_after_combine10')(P5_out)

        P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)

        if normal_fusion:
            P6_out = Add(name='fpn_cells/cell_/fnode6/add')([P6_in, P6_td, P5_D])
        else:
            P6_out = weightAdd(name='fpn_cells/cell_/fnode6/add')([P6_in, P6_td, P5_D])

        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)

        if normal_fusion:
            P7_out = Add(name='fpn_cells/cell_/fnode7/add')([P7_in, P6_D])
        else:
            P7_out = weightAdd(name='fpn_cells/cell_/fnode7/add')([P7_in, P6_D])

        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode7/op_after_combine12')(P7_out)

        print('p3', P3_out)
        print('p4', P4_td)
        print('p5', P5_td)
        print('p6', P6_td)
        print('p7', P7_out)
        return P3_out, P4_td, P5_td, P6_td, P7_out

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features

        P7_U = UpSampling2D()(P7_in)

        if normal_fusion:
            P6_td = Add(name=f'fpn_cells/cell_{times}/fnode0/add')([P6_in, P7_U])
        else:
            P6_td = weightAdd(name=f'fpn_cells/cell_{times}/fnode0/add')([P6_in, P7_U])

        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{times}/fnode0/op_after_combine5')(P6_td)
        P6_U = UpSampling2D()(P6_td)


        if normal_fusion:
            P5_td = Add(name=f'fpn_cells/cell_{times}/fnode1/add')([P5_in, P6_U])
        else:
            P5_td = weightAdd(name=f'fpn_cells/cell_{times}/fnode1/add')([P5_in, P6_U])

        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{times}/fnode1/op_after_combine6')(P5_td)
        P5_U = UpSampling2D()(P5_td)

        if normal_fusion:
            P4_td = Add(name=f'fpn_cells/cell_{times}/fnode2/add')([P4_in, P5_U])
        else:
            P4_td = weightAdd(name=f'fpn_cells/cell_{times}/fnode2/add')([P4_in, P5_U])

        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{times}/fnode2/op_after_combine7')(P4_td)
        P4_U = UpSampling2D()(P4_td)

        if normal_fusion:
            P3_out = Add(name=f'fpn_cells/cell_{times}/fnode3/add')([P3_in, P4_U])
        else:
            P3_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode3/add')([P3_in, P4_U])

        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode3/op_after_combine8')(P3_out)
        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)

        if normal_fusion:
            P4_out = Add(name=f'fpn_cells/cell_{times}/fnode4/add')([P4_in, P4_td, P3_D])
        else:
            P4_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode4/add')([P4_in, P4_td, P3_D])

        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode4/op_after_combine9')(P4_out)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

        if normal_fusion:
            P5_out = Add(name=f'fpn_cells/cell_{times}/fnode5/add')([P5_in, P5_td, P4_D])
        else:
            P5_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode5/add')([P5_in, P5_td, P4_D])

        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode5/op_after_combine10')(P5_out)

        P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)

        if normal_fusion:
            P6_out = Add(name=f'fpn_cells/cell_{times}/fnode6/add')([P6_in, P6_td, P5_D])
        else:
            P6_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode6/add')([P6_in, P6_td, P5_D])

        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)

        if normal_fusion:
            P7_out = Add(name=f'fpn_cells/cell_{times}/fnode7/add')([P7_in, P6_D])
        else:
            P7_out = weightAdd(name=f'fpn_cells/cell_{times}/fnode7/add')([P7_in, P6_D])

        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{times}/fnode7/op_after_combine12')(P7_out)

        return P3_out, P4_td, P5_td, P6_td, P7_out


def csnet_extra_model(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], regularization=5e-4):
    source_layers = []
    base = create_efficientNet(base_model_name, pretrained, IMAGE_SIZE)

    layer_names = GET_EFFICIENT_NAME[base_model_name]
    # get extra layer
    #efficient_conv75 = base.get_layer('block2b_add').output  # 75 75 24
    p3 = base.get_layer(layer_names[0]).output # 64 64 40
    # p4 = base.get_layer('block4c_add').output
    p5 = base.get_layer(layer_names[1]).output # 32 32 112
    # p6 = base.get_layer('block6d_add').output
    p7 = base.get_layer(layer_names[2]).output # 16 16 320

    features = p3, p5, p7
    for i in range(3):
        print("times i : ", i+1)
        features = build_FPN(features, 64, times=i, normal_fusion=True)



    extra_p8 = SeparableConvBlock(num_channels=64, kernel_size=3, strides=2,
                               name='extra/p8/dwconv')(features[4])
    extra_p8 = Activation(lambda x: tf.nn.swish(x))(extra_p8)

    extra_p9 = SeparableConvBlock(num_channels=64, kernel_size=3, strides=2,
                               name='extra/p9/dwconv')(extra_p8)
    extra_p9 = Activation(lambda x: tf.nn.swish(x))(extra_p9)

    # conv1 = extraMBConv(features[4], 'same', 'conv3_to_conv1_1')
    # conv1 = extraMBConv(conv1, 'valid', 'conv3_to_conv1_2')
    #
    # conv0 = extraMBConv(conv1, 'same', 'conv1_to_conv0_1')
    # conv0 = extraMBConv(conv0, 'same', 'conv0_1_to_conv0_2', (2, 2))

    # predict features
    source_layers.append(features[0])
    source_layers.append(features[1])
    source_layers.append(features[2])
    source_layers.append(features[3])
    source_layers.append(features[4])
    source_layers.append(extra_p8)
    source_layers.append(extra_p9)
    print(features[0])
    print(features[1])
    print(features[2])
    print(features[3])
    print(features[4])
    print(extra_p8)
    print(extra_p9)

    return base.input, source_layers