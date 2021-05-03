from tensorflow.keras.layers import Conv2D, BatchNormalization,\
    Activation, Dense, Concatenate, Flatten, Reshape, Dropout,\
    SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from keras.engine.topology import Layer
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
import tensorflow as tf

# l2 normalize
class Normalize(Layer):
    def __init__(self, scale=20, **kwargs):
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name=self.name+'_gamma', 
                                     shape=(input_shape[-1],),
                                     initializer=Constant(self.scale), 
                                     trainable=True)
        super(Normalize, self).build(input_shape)
        
    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, axis=-1)

# CLASSIFIER BUILD!
def create_classifier(source_layers, num_priors, normalizations, num_classes=21, classifier_times=3):
    mbox_conf = []
    mbox_loc = []
    for i, layer in enumerate(source_layers):
        # source_layers
        # name='block3b_add/add_1:0  shape=(batch, 38, 38, 40)
        # name='block5c_add/add_1:0 shape=(batch, 19, 19, 112)
        # name='block7a_project_bn/cond_1/Identity:0' shape=(batch, 10, 10, 320)
        # name='activation_1/Relu:0' shape=(batch, 5, 5, 256)
        # name='activation_3/Relu:0' shape=(batch, 3, 3, 256)
        # name='activation_5/Relu:0' shape=(batch, 1, 1, 256)
        x = layer
        # name = x.name.split('/')[0] # name만 추출 (ex: block3b_add)
        name = x.name.split(':')[0] # name만 추출 (ex: block3b_add)

        # <<< reduce norm
        if normalizations is not None and normalizations[i] > 0:
           x = Normalize(normalizations[i], name=name + '_norm')(x)
           #print('norm_feature : '+x.name)

        # x = activation_5/Relu:0, shape=(Batch, 1, 1, 256)
        # print("start_multibox_head.py")
        # print("num_priors[i]",num_priors[i]) #6 (첫 번째 38,38일 경우)
        # print("num_classes",num_classes) #21
        # print("num_priors[i] * num_classes",num_priors[i] * num_classes) # 126

        ## original ----
        # x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same', kernel_regularizer=l2(5e-4) ,name= name + '_mbox_conf')(x)
        # x1 = SeparableConv2D(num_priors[i] * num_classes, 3, padding='same', use_bias=False, kernel_regularizer=l2(5e-4), name= name + '_mbox_conf')(x)
        x1 = SeparableConv2D(num_priors[i] * num_classes, 3, padding='same',
                             depthwise_initializer=initializers.VarianceScaling(),
                             pointwise_initializer=initializers.VarianceScaling(),
                             name= name + '_mbox_conf_1')(x)

        # for cls_times in range(classifier_times-1):
        #     #print('cls_times', cls_times)
        #     x1 = SeparableConv2D(num_priors[i] * num_classes, 3, padding='same',
        #                          depthwise_initializer=initializers.VarianceScaling(),
        #                          pointwise_initializer=initializers.VarianceScaling(),
        #                          name= name + '_mbox_conf_'+str(cls_times+2))(x1)

        x1 = Flatten(name=name + '_mbox_conf_flat')(x1)


        # x1 = activation_b5_mbox_conf_flat/Reshape:0 , shape=(Batch , 84)
        mbox_conf.append(x1)

        # x2 = Conv2D(num_priors[i] * 4, 3, padding='same', kernel_regularizer=l2(5e-4) ,name= name + '_mbox_loc')(x)
        # x2 = SeparableConv2D(num_priors[i] * 4, 3, padding='same', use_bias=False, kernel_regularizer=l2(5e-4),name= name + '_mbox_loc')(x)
        x2 = SeparableConv2D(num_priors[i] * 4, 3, padding='same',
                             depthwise_initializer=initializers.VarianceScaling(),
                             pointwise_initializer=initializers.VarianceScaling(),
                             name= name + '_mbox_loc_1')(x)

        # for loc_times in range(classifier_times - 1):
        #     x2 = SeparableConv2D(num_priors[i] * 4, 3, padding='same',
        #                          depthwise_initializer=initializers.VarianceScaling(),
        #                          pointwise_initializer=initializers.VarianceScaling(),
        #                          name= name + '_mbox_loc_'+str(loc_times+2))(x2)

        x2 = Flatten(name=name + '_mbox_loc_flat')(x2)
        # x2 = activation_b5_mbox_loc_flat/Reshape:0 , shape=(Batch , 16)
        mbox_loc.append(x2)

    # mbox_loc/concat:0 , shape=(Batch, 34928)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')(mbox_loc)
    # mbox_loc_final/Reshape:0, shape=(Batch, 8732, 4)
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    # mobx_conf/concat:0, shape=(Batch, 183372)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')(mbox_conf)
    # mbox_conf_logits/Reshape:0, shape=(None, 8732, 21)
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    # mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    # predictions/concat:0, shape=(Batch, 8732, 25)
    predictions = Concatenate(axis=2, name='predictions', dtype=tf.float32)([mbox_loc, mbox_conf])

    return predictions
