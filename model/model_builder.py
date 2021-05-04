import tensorflow as tf
from tensorflow import keras
from model.model import csnet_extra_model
from model.model_classifer import create_classifier

# train.py에서 priors를 변경하면 여기도 수정해야함
def model_build(model_mode, base_model_name, pretrained=True, backbone_trainable=True, image_size=[512, 512], normalizations=[20, 20, 20, -1, -1], num_priors=[3, 3, 3, 3, 3]):
    if model_mode == 'voc':
        classes = 21
    elif model_mode == 'coco':
        classes = 81

    inputs, source_layers, classifier_times = csnet_extra_model(base_model_name, pretrained, image_size, backbone_trainable=backbone_trainable)
    output = create_classifier(source_layers, num_priors, normalizations, classes, classifier_times)
    model = keras.Model(inputs, outputs=output)
    return model

