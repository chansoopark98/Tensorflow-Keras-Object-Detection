import tensorflow as tf
from tensorflow import keras
from model.model import csnet_extra_model
from model.coco_classifer import create_classifier

# train.py에서 priors를 변경하면 여기도 수정해야함

def ssd(base_model_name, pretrained=True, image_size=[300, 300], normalizations=[20, 20, 20, -1, -1, -1], num_priors=[4,6,6,6,4,4]):
    inputs, source_layers = csnet_extra_model(base_model_name, pretrained, image_size)
    output = create_classifier(source_layers, num_priors, normalizations)
    model = keras.Model(inputs, outputs=output)
    return model

