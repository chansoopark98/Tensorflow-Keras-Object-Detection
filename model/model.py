import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Add, Activation, Dropout ,BatchNormalization,  UpSampling2D,\
    SeparableConv2D, MaxPooling2D, Concatenate, DepthwiseConv2D, GlobalAveragePooling2D, Reshape, multiply, add
from functools import reduce

NUM_CHANNELS = [64, 64, 88, 112, 160, 224, 288, 288, 288]
#NUM_CHANNELS = [48, 64, 88, 112, 160, 244, 288, 288]
FPN_TIMES = [1, 3, 4, 5, 6, 7, 7, 7, 7]
CLS_TIEMS = [1, 3, 3, 3, 4, 4, 4, 4, 4]


MOMENTUM = 0.997
EPSILON = 1e-4

GET_EFFICIENT_NAME = {
    'B0-tiny': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B0': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B1': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B2': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B3': ['block3c_add', 'block5e_add', 'block7b_add'],
    'B4': ['block3d_add', 'block5f_add', 'block7b_add'],
    'B5': ['block3e_add', 'block5g_add', 'block7c_add'],
    'B6': ['block3f_add', 'block5h_add', 'block7c_add'],
    'B7': ['block3g_add', 'block5j_add', 'block7d_add'],
}

MODEL_NAME = {
    'B0-tiny': 0,
    'B0': 1,
    'B1': 2,
    'B2': 3,
    'B3': 4,
    'B4': 5,
    'B5': 6,
    'B6': 7,
    'B7': 7,
}


def remove_dropout(model):
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    del model

    return model_copy



def create_efficientNet(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], trainable=True):
    if pretrained is False:
        weights = None



    else:
        weights = "imagenet"

    if base_model_name == 'B0' or 'B0-tiny':
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
    base.trainable = trainable


    return base




def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=name+'/conv')
    f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

def build_BiFPN(features, num_channels=64 , id=0, resize=False, bn_trainable=True):
    if resize:
        padding = 'valid'
    else:
        padding = 'same'

    if id == 0:


        C3, C4, C5 = features
        P3_in = C3 # 36x36
        P4_in = C4 # 18x18
        P5_in = C5 # 9x9

        P6_in = Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,  trainable=bn_trainable, name='resample_p6/bn')(P6_in)

        # padding
        P6_in = MaxPooling2D(pool_size=3, strides=2, padding=padding, name='resample_p6/maxpool')(P6_in) # 4x4

        P7_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in) # 2x2


        if resize:
            P7_U = tf.image.resize(P7_in, (P6_in.shape[1:3]))
        else:
            P7_U = UpSampling2D()(P7_in) # 2x2 to 4x4

        P6_td = Add(name='fpn_cells/cell_/fnode0/add')([P6_in, P7_U])
        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                            name='fpn_cells/cell_/fnode1/resample_0_2_6/bn')(P5_in_1)

        if resize:
            P6_U = tf.image.resize(P6_td, (P5_in_1.shape[1:3]))
        else:
            P6_U = UpSampling2D()(P6_td)

        P5_td = Add(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U]) # 9x9
        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode2/resample_0_1_7/conv2d')(P4_in) # 18x18
        P4_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                            name='fpn_cells/cell_/fnode2/resample_0_1_7/bn')(P4_in_1)

        P5_U = UpSampling2D()(P5_td)
        P4_td = Add(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U]) # 18x18
        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode2/op_after_combine7')(P4_td)
        P3_in = Conv2D(num_channels, kernel_size=1, padding='same',
                              name='fpn_cells/cell_/fnode3/resample_0_0_8/conv2d')(P3_in) # 36x36
        P3_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                          name=f'fpn_cells/cell_/fnode3/resample_0_0_8/bn')(P3_in)

        P4_U = UpSampling2D()(P4_td) # 18x18 to 36x36
        P3_out = Add(name='fpn_cells/cell_/fnode3/add')([P3_in, P4_U])
        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                            name='fpn_cells/cell_/fnode4/resample_0_1_9/bn')(P4_in_2)

        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = Add(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                            name='fpn_cells/cell_/fnode5/resample_0_2_10/bn')(P5_in_2)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = Add(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D]) # 9x9
        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode5/op_after_combine10')(P5_out)

        # padding
        P5_D = MaxPooling2D(pool_size=3, strides=2, padding=padding)(P5_out) # 9x9 to 4x4

        P6_out = Add(name='fpn_cells/cell_/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = Add(name='fpn_cells/cell_/fnode7/add')([P7_in, P6_D])
        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode7/op_after_combine12')(P7_out)


        return [P3_out, P4_td, P5_td, P6_td, P7_out]

    else:

        P3_in, P4_in, P5_in, P6_in, P7_in = features



        if resize:
            P7_U = tf.image.resize(P7_in, (P6_in.shape[1:3]))
        else:
            P7_U = UpSampling2D()(P7_in) # 2x2 to 4x4

        P6_td = Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)

        if resize:
            P6_U = tf.image.resize(P6_td, (P5_in.shape[1:3]))
        else:
            P6_U = UpSampling2D()(P6_td) # 4x4 to 9x9

        P5_td = Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U]) # 9x9
        P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = UpSampling2D()(P5_td) # 9x9 to 18x18
        P4_td = Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U]) # 18x18
        P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = UpSampling2D()(P4_td) # 18x18 to 36x36
        P3_out = Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out) # 36x36 to 18x18
        P4_out = Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
        P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out) # 18x18 to 9x9
        P5_out = Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        # padding
        P5_D = MaxPooling2D(pool_size=3, strides=2, padding=padding)(P5_out)  # 9x9 to 4x4

        P6_out = Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)



        return [P3_out, P4_td, P5_td, P6_td, P7_out]

def build_tiny_BiFPN(features, num_channels=64 , id=0, resize=True, bn_trainable=True):
    if resize:
        padding = 'valid'
    else:
        padding = 'same'


    C3, C4, C5 = features
    P3_in = C3 # 36x36
    P4_in = C4 # 18x18
    P5_in = C5 # 9x9

    P6_in = Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
    P6_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,  trainable=bn_trainable, name='resample_p6/bn')(P6_in)

    # padding
    P6_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in) # 5x5

    P7_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in) # 3x3


    if resize:
        P7_U = tf.image.resize(P7_in, (P6_in.shape[1:3])) # 5x5
    else:
        P7_U = UpSampling2D()(P7_in) # 2x2 to 4x4

    P6_td = Add(name='fpn_cells/cell_/fnode0/add')([P6_in, P7_U]) # 5x5
    P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
    P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name='fpn_cells/cell_/fnode0/op_after_combine5')(P6_td)
    P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same', # 10x10
                            name='fpn_cells/cell_/fnode1/resample_0_2_6/conv2d')(P5_in)
    P5_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                        name='fpn_cells/cell_/fnode1/resample_0_2_6/bn')(P5_in_1)

    if resize:
        P6_U = tf.image.resize(P6_td, (P5_in_1.shape[1:3])) # 10x10
    else:
        P6_U = UpSampling2D()(P6_td)

    P5_td = Add(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U]) # 9x9
    P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
    P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name='fpn_cells/cell_/fnode1/op_after_combine6')(P5_td)
    P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                            name='fpn_cells/cell_/fnode2/resample_0_1_7/conv2d')(P4_in) # 18x18
    P4_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                        name='fpn_cells/cell_/fnode2/resample_0_1_7/bn')(P4_in_1)

    #P5_U = UpSampling2D()(P5_td) # 20x20
    P5_U = tf.image.resize(P5_td, (P4_in_1.shape[1:3]))  # 5x5
    P4_td = Add(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U]) # 18x18
    P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
    P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name='fpn_cells/cell_/fnode2/op_after_combine7')(P4_td)
    P3_in = Conv2D(num_channels, kernel_size=1, padding='same',
                          name='fpn_cells/cell_/fnode3/resample_0_0_8/conv2d')(P3_in) # 36x36
    P3_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                      name=f'fpn_cells/cell_/fnode3/resample_0_0_8/bn')(P3_in)

    P4_U = UpSampling2D()(P4_td) # 18x18 to 36x36
    P3_out = Add(name='fpn_cells/cell_/fnode3/add')([P3_in, P4_U])
    P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
    P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name='fpn_cells/cell_/fnode3/op_after_combine8')(P3_out)
    P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                            name='fpn_cells/cell_/fnode4/resample_0_1_9/conv2d')(P4_in)
    P4_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                        name='fpn_cells/cell_/fnode4/resample_0_1_9/bn')(P4_in_2)

    P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
    P4_out = Add(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td, P3_D])
    P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
    P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name='fpn_cells/cell_/fnode4/op_after_combine9')(P4_out)

    P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                            name='fpn_cells/cell_/fnode5/resample_0_2_10/conv2d')(P5_in)
    P5_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
                                        name='fpn_cells/cell_/fnode5/resample_0_2_10/bn')(P5_in_2)

    P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
    P5_out = Add(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D]) # 9x9
    P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
    P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name='fpn_cells/cell_/fnode5/op_after_combine10')(P5_out)

    # padding
    P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out) # 9x9 to 4x4

    P6_out = Add(name='fpn_cells/cell_/fnode6/add')([P6_in, P6_td, P5_D])
    P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
    P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name='fpn_cells/cell_/fnode6/op_after_combine11')(P6_out)

    P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
    P7_out = Add(name='fpn_cells/cell_/fnode7/add')([P7_in, P6_D])
    P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
    P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name='fpn_cells/cell_/fnode7/op_after_combine12')(P7_out)


    return [P3_out, P4_td, P5_td, P6_td, P7_out]

def csnet_extra_model(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], backbone_trainable=True):

    if backbone_trainable == True:
        bn_trainable = True
    else:
        bn_trainable = True

    print("backbone_trainable", backbone_trainable)
    print("bn_trainable", bn_trainable)
    source_layers = []
    base = create_efficientNet(base_model_name, pretrained, IMAGE_SIZE, trainable=backbone_trainable)

    layer_names = GET_EFFICIENT_NAME[base_model_name]

    # get extra layer
    p3 = base.get_layer(layer_names[0]).output # 64 64 40
    p5 = base.get_layer(layer_names[1]).output # 32 32 112
    p7 = base.get_layer(layer_names[2]).output # 16 16 320

    features = [p3, p5, p7]
    print(features)
    if base_model_name == 'B0':
        feature_resize = False
    else:
        feature_resize = True


    if base_model_name == 'B0-tiny':
            features = build_tiny_BiFPN(features=features, num_channels=NUM_CHANNELS[MODEL_NAME[base_model_name]],
                                   id=0, resize=feature_resize, bn_trainable=bn_trainable)
    else:
        for i in range(FPN_TIMES[MODEL_NAME[base_model_name]]):
            print("times", i)
            features = build_BiFPN(features=features, num_channels=NUM_CHANNELS[MODEL_NAME[base_model_name]],
                                   id=i, resize=feature_resize, bn_trainable=bn_trainable)

    # predict features
    source_layers.append(features[0])
    source_layers.append(features[1])
    source_layers.append(features[2])
    source_layers.append(features[3])
    source_layers.append(features[4])

    return base.input, source_layers, CLS_TIEMS[MODEL_NAME[base_model_name]]

"""CSNet-tiny hyper parameters"""

width_coefficient = 1.0
depth_divisor = 1.0
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}
_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None

backend = None
layers = None
models = None
keras_utils = None

bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1


def round_filters(filters, width_coefficient, depth_divisor):
    """너비 승수를 기준으로 한 필터 수를 반올림"""
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # 내림이 10% 이상 내려가지 않도록 함
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


#activation = lambda x: tf.nn.swish(x)
activation = 'relu'

def tiny_stem_block(input):
    x = Conv2D(32, 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(input)
    x = BatchNormalization(axis=bn_axis, name='stem_bn')(x)

    x = Activation(activation)(x)
    return x



def tiny_expand_conv_block(input, init_channel):
    x = SeparableConv2D(filters=init_channel*3, kernel_size=3, strides=1,  padding='same', use_bias=True)(input)
    x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=True)(x)
    x = Activation(lambda x: tf.nn.swish(x))(x)

    x = Conv2D(filters=init_channel, kernel_size=1, strides=1,  padding='same', use_bias=True)(x)
    #x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=True)(x)
    x = Activation(activation)(x)
    return x


def csnet_tiny_block(inputs, input_filters, output_filters,
                     expand_ratio, kernel_size, strides, has_se, drop_rate=0.2):
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1


    # Expansion phase
    filters = input_filters * expand_ratio
    x = Conv2D(filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      )(inputs)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation(activation)(x)

    # Depthwise Convolution
    x = DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               )(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation(activation)(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(input_filters * 0.25))
        se_tensor = GlobalAveragePooling2D()(x)

        target_shape = (1, 1, filters) if tf.keras.backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = Reshape(target_shape)(se_tensor)
        se_tensor = Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  )(se_tensor)
        se_tensor = Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  )(se_tensor)
        x = multiply([x, se_tensor])

    # Output phase
    x = Conv2D(output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      )(x)
    x = BatchNormalization(axis=bn_axis)(x)
    if strides != 2:
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        )(x)
        x = Concatenate()([x, inputs])

    return x

def tiny_csnet(base_model_name, IMAGE_SIZE=[300, 300]):
    # CSNet-tiny inputs
    input = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # 16, 24, 40, 80, 112, 192, 320

    # STEM block
    stem = tiny_stem_block(input) # 150x150


    # conv1
    conv1 = csnet_tiny_block(inputs=stem, input_filters=32, output_filters=48, expand_ratio=1,
                             kernel_size=3, strides=1, has_se=True, drop_rate=0)
    conv1 = csnet_tiny_block(inputs=conv1, input_filters=32, output_filters=48, expand_ratio=3,
                             kernel_size=3, strides=1, has_se=True, drop_rate=0)


    # conv2
    conv2 = csnet_tiny_block(inputs=conv1, input_filters=48, output_filters=56, expand_ratio=3,
                             kernel_size=3, strides=2, has_se=True, drop_rate=0)
    conv2 = csnet_tiny_block(inputs=conv2, input_filters=48, output_filters=56, expand_ratio=3,
                             kernel_size=3, strides=1, has_se=True, drop_rate=0)


    # conv3
    conv3 = csnet_tiny_block(inputs=conv2, input_filters=56, output_filters=72, expand_ratio=3,
                             kernel_size=3, strides=2, has_se=True, drop_rate=0)
    conv3 = csnet_tiny_block(inputs=conv3, input_filters=56, output_filters=72, expand_ratio=3,
                             kernel_size=5, strides=1, has_se=True, drop_rate=0)


    # conv4
    conv4 = csnet_tiny_block(inputs=conv3, input_filters=72, output_filters=88, expand_ratio=3,
                             kernel_size=3, strides=2, has_se=True, drop_rate=0)
    conv4 = csnet_tiny_block(inputs=conv4, input_filters=72, output_filters=88, expand_ratio=3,
                             kernel_size=5, strides=1, has_se=True, drop_rate=0)



    # conv5
    conv5 = csnet_tiny_block(inputs=conv4, input_filters=88, output_filters=112, expand_ratio=3,
                             kernel_size=3, strides=2, has_se=True, drop_rate=0)
    conv5 = csnet_tiny_block(inputs=conv5, input_filters=88, output_filters=112, expand_ratio=3,
                             kernel_size=5, strides=1, has_se=True, drop_rate=0)



    # conv6
    conv6 = csnet_tiny_block(inputs=conv5, input_filters=112, output_filters=128, expand_ratio=3,
                             kernel_size=3, strides=2, has_se=True, drop_rate=0)
    conv6 = csnet_tiny_block(inputs=conv6, input_filters=112, output_filters=128, expand_ratio=3,
                             kernel_size=5, strides=1, has_se=True, drop_rate=0)

    # conv7
    conv7 = csnet_tiny_block(inputs=conv6, input_filters=128, output_filters=144, expand_ratio=3,
                             kernel_size=3, strides=2, has_se=True, drop_rate=0)
    conv7 = csnet_tiny_block(inputs=conv7, input_filters=128, output_filters=144, expand_ratio=3,
                             kernel_size=3, strides=1, has_se=True, drop_rate=0)


    outputs = [conv3, conv4, conv5, conv6, conv7]
    return input, outputs
