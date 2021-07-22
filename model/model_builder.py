from tensorflow import keras
from model.model import csnet_extra_model, tiny_csnet
from model.model_classifer import create_classifier, tiny_classifier
from model.seg_model import csnet_seg_model


# train.py에서 priors를 변경하면 여기도 수정해야함
def model_build(model_mode, base_model_name, pretrained=True, backbone_trainable=True, image_size=[512, 512], normalizations=[20, 20, 20, -1, -1], num_priors=[3, 3, 3, 3, 3]):

    if model_mode == 'voc':
        classes = 21
    else:
        classes = 81

    if base_model_name == 'CSNet-tiny':
        inputs, source_layers = tiny_csnet(image_size)
        output = tiny_classifier(source_layers, num_priors, classes)
        model = keras.Model(inputs, output)
        return model
    else:

        inputs, source_layers, classifier_times = csnet_extra_model(base_model_name, pretrained, image_size, backbone_trainable=backbone_trainable)
        output = create_classifier(source_layers, num_priors, normalizations, classes, classifier_times)
        model = keras.Model(inputs, outputs=output)
        return model


def seg_model_build(base_model_name, pretrained, image_size):
    #input, output = csnet_seg_model(base_model_name, pretrained, image_size) \
    input, output =  csnet_seg_model(weights=None, input_tensor=None, input_shape=(image_size[0], image_size[1], 3), classes=19, OS=16)
    model = keras.Model(input, output)
    return model
