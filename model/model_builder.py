from tensorflow.keras.layers import Concatenate, Flatten, Reshape, SeparableConv2D, Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, VarianceScaling
from tensorflow.keras.regularizers import L2
import tensorflow as tf
from tensorflow.keras.models import Model






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
    def __init__(self, image_size: tuple = (300, 300), num_classes: int = 21,
     use_weight_decay: bool = False, weight_decay: float = 0.00001):
        """
        Args:
            image_size         (tuple) : Model input resolution ([H, W])
            num_classes        (int)   : Number of classes to classify 
                                         (must be equal to number of last filters in the model)
            use_weight_decay   (bool)  : Use weight decay.
            weight_decay       (float) : Weight decay value.
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_weight_decay = use_weight_decay
        self.weight_decay = weight_decay
        self.kernel_initializer = VarianceScaling(scale=2.0, mode="fan_out",
                                                  distribution="truncated_normal")
        self.normalize = [20, 20, 20, -1, -1, -1]
        self.num_priors = [4, 6, 6, 6, 4, 4]


    def build_model(self, model_name: str) -> Model:
        if model_name == 'mobilenetv2':
            from .model_zoo.mobileNetV2_ssd import MobileNetV2
            model = MobileNetV2(image_size=self.image_size, pretrained="imagenet")
        elif model_name == 'mobilenetv3s':
            from .model_zoo.mobileNetV3small import MobileNetV3S
            model = MobileNetV3S(image_size=self.image_size, pretrained="imagenet")
        elif model_name == 'mobilenetv3l':
            from .model_zoo.mobileNetV3large import MobileNetV3L
            model = MobileNetV3L(image_size=self.image_size, pretrained="imagenet")
        elif model_name == 'efficient_lite_v0':
            from .model_zoo.EfficientNetLiteB0 import EfficientLiteB0
            model = EfficientLiteB0(image_size=self.image_size, pretrained="imagenet")
        elif model_name == 'efficientv2b0':
            from .model_zoo.EffcientNetV2B0 import EfficientNetV2B0
            self.normalize = [20, 20, 20, -1, -1, -1]
            model = EfficientNetV2B0(image_size=self.image_size, pretrained="imagenet")
        elif model_name == 'efficientv2b3':
            from .model_zoo.EfficientNetV2B3 import EfficientNetV2B3
            model = EfficientNetV2B3(image_size=self.image_size, pretrained="imagenet")
        else:
            raise NotImplementedError('Your input model_name is not implemented')
        
        model_input, detector_output = model.build_extra_layer()
        
        model_output = self.create_classifier(source_layers=detector_output,
                               num_priors=self.num_priors, normalizations=self.normalize)


        model = Model(inputs=model_input, outputs=model_output)

        if self.use_weight_decay:
            for layer in model.layers:        
                if isinstance(layer, Conv2D):
                    layer.add_loss(lambda layer=layer: L2(self.weight_decay)(layer.kernel))
                elif isinstance(layer, SeparableConv2D):
                    layer.add_loss(lambda layer=layer: L2(self.weight_decay)(layer.depthwise_kernel))
                    layer.add_loss(lambda layer=layer: L2(self.weight_decay)(layer.pointwise_kernel))
                elif isinstance(layer, DepthwiseConv2D):
                    layer.add_loss(lambda layer=layer: L2(self.weight_decay)(layer.depthwise_kernel)) 

                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.add_loss(lambda layer=layer: L2(self.weight_decay)(layer.bias))

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
            # x1 = Conv2D(num_priors[i] * self.num_classes, 3, padding='same',
            #             kernel_initializer=self.kernel_initializer,
            #             name=name + '_mbox_conf_1')(x)
            x1 = Flatten(name=name + '_mbox_conf_flat')(x1)
            mbox_conf.append(x1)

            
            x2 = SeparableConv2D(num_priors[i] * 4, 3, padding='same',
                                depthwise_initializer=self.kernel_initializer,
                                pointwise_initializer=self.kernel_initializer,
                                name= name + '_mbox_loc_1')(x)
            # x2 = Conv2D(num_priors[i] * 4, 3, padding='same',
            #                     kernel_initializer=self.kernel_initializer,
            #                     name= name + '_mbox_loc_1')(x)
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