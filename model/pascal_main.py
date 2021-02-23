import tensorflow as tf
from tensorflow import keras
from model.model import csnet_extra_model
from model.pascal_classifer import create_classifier

# train.py에서 priors를 변경하면 여기도 수정해야함

def ssd(model_mode, base_model_name, pretrained=True, image_size=[384, 384], normalizations=[20, 20, 20, -1, -1, -1], num_priors=[4,6,6,6,4,4]):
    if model_mode == 'voc':
        classes = 21
    else:
        classes = 81

    inputs, source_layers = csnet_extra_model(base_model_name, pretrained, image_size)
    output = create_classifier(source_layers, num_priors, normalizations, num_classes=classes)
    model = keras.Model(inputs, outputs=output)
    return model

