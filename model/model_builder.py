
from tensorflow.keras.layers import Concatenate, Flatten, Reshape, SeparableConv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, VarianceScaling
import tensorflow as tf
from tensorflow.keras.models import Model
from .model_zoo.mobileNet_ssd import MobileNetV2
from .model_zoo.EffcientNetV2B0 import EfficientNetV2B0
from .model_zoo.EfficientNetV2B3 import EfficientNetV2B3
import tensorflow.keras.backend as K

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

    def get_config(self):
        config = super().get_config().copy()
        return config


class ModelBuilder():
    def __init__(self, image_size: tuple = (300, 300), num_classes: int = 21):
        """
        Args:
            image_size  (tuple) : Model input resolution ([H, W])
            num_classes (int)   : Number of classes to classify 
                                  (must be equal to number of last filters in the model)
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.kernel_initializer = VarianceScaling(scale=2.0, mode="fan_out",
                                                  distribution="truncated_normal")
        self.normalize = [20, 20, 20, -1, -1, -1]
        self.num_priors = [4, 6, 6, 6, 4, 4]


    def build_model(self, model_name: str) -> Model:
        if model_name == 'mobilenetv2':
            model = MobileNetV2(image_size=self.image_size, pretrained="imagenet")
        elif model_name == 'efficientv2b0':
            model = EfficientNetV2B0(image_size=self.image_size, pretrained="imagenet")
        elif model_name == 'efficientv2b3':
            model = EfficientNetV2B3(image_size=self.image_size, pretrained="imagenet")
        else:
            raise NotImplementedError('Your input model_name is not implemented')
        
        model_input, detector_output = model.build_extra_layer()
        
        model_output = self.create_classifier(source_layers=detector_output,
                               num_priors=self.num_priors, normalizations=self.normalize)

        model = Model(inputs=model_input, outputs=model_output)

        return model


    def create_classifier(self, source_layers, num_priors, normalizations):
        mbox_conf = []
        mbox_loc = []
        for i, x in enumerate(source_layers):
            name = x.name.split(':')[0] # name만 추출 (ex: block3b_add)

            if normalizations is not None and normalizations[i] > 0:
                x = Normalize(normalizations[i], name=name + '_norm')(x)

            x1 = SeparableConv2D(num_priors[i] * self.num_classes, 3, padding='same',
                                depthwise_initializer=self.kernel_initializer,
                                pointwise_initializer=self.kernel_initializer,
                                name= name + '_mbox_conf_1')(x)
            x1 = Flatten(name=name + '_mbox_conf_flat')(x1)
            mbox_conf.append(x1)

            
            x2 = SeparableConv2D(num_priors[i] * 4, 3, padding='same',
                                depthwise_initializer=self.kernel_initializer,
                                pointwise_initializer=self.kernel_initializer,
                                name= name + '_mbox_loc_1')(x)
            x2 = Flatten(name=name + '_mbox_loc_flat')(x2)
            mbox_loc.append(x2)

        # mbox_loc/concat:0 , shape=(Batch, 34928)
        mbox_loc = Concatenate(axis=1, name='mbox_loc')(mbox_loc)
        # mbox_loc_final/Reshape:0, shape=(Batch, 8732, 4)
        mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

        # mobx_conf/concat:0, shape=(Batch, 183372)
        mbox_conf = Concatenate(axis=1, name='mbox_conf')(mbox_conf)
        # mbox_conf_logits/Reshape:0, shape=(None, 8732, 21)
        mbox_conf = Reshape((-1, self.num_classes), name='mbox_conf_logits')(mbox_conf)
        # mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

        # predictions/concat:0, shape=(Batch, 8732, 25)
        predictions = Concatenate(axis=2, name='predictions', dtype=tf.float32)([mbox_loc, mbox_conf])

        return predictions