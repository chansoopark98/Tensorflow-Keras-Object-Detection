from utils.priors import *

iou_threshold = 0.5 # 0.5
center_variance = 0.1 # 0.1
size_variance = 0.2 # 0.2


MODEL_INPUT_SIZE = {
    'B0': 512,
    'B1': 544,
    'B2': 576, # 576
    'B3': 704,
    'B4': 768,
    'B5': 832,
    'B6': 896,
    'B7': 960
}

class TrainHyperParams:
    def __init__(self):
        self.optimizer_name = 'sgd'
        self.weight_decay = 0.0005
        self.learning_rate = 0.001
        self.lr_decay_steps = 200
        self.epochs = 200



# Spec(32, 16, BoxSizes(51, 122), [2, 3]),
# Spec(16, 32, BoxSizes(122, 189), [2, 3]),
# Spec(8, 64, BoxSizes(230, 296), [2, 3]),
# Spec(4, 128, BoxSizes(307, 389), [2]),
# Spec(2, 256, BoxSizes(389, 460), [2]),


# Spec(64, 8, BoxSizes(25, 51), [2]),
# Spec(32, 16, BoxSizes(51, 122), [2]),
# Spec(16, 32, BoxSizes(122, 189), [2]),
# Spec(8, 64, BoxSizes(230, 296), [2]),
# Spec(4, 128, BoxSizes(307, 389), [2]),
#
# Spec(64, 8, BoxSizes(25, 51), [2]),
# Spec(32, 16, BoxSizes(51, 122), [2, 3]),
# Spec(16, 32, BoxSizes(122, 189), [2, 3]),
# Spec(8, 64, BoxSizes(230, 296), [2, 3]),
# Spec(4, 128, BoxSizes(307, 389), [2]),

# # 0504 voc b0 mAP 81.6%
# Spec(64, 8, BoxSizes(25, 51), [2]),
# Spec(32, 16, BoxSizes(51, 122), [2]),
# Spec(16, 32, BoxSizes(122, 189), [2]),
# Spec(8, 64, BoxSizes(230, 296), [2]),
# Spec(4, 128, BoxSizes(307, 389), [2]),
def set_priorBox(model_name):
    if model_name == 'B0':
        return [
            Spec(64, 8, BoxSizes(20, 25), [2]), # 0.039
            Spec(32, 16, BoxSizes(41, 51), [2]), # 0.099
            Spec(16, 32, BoxSizes(92, 112), [2]), # 0.238 -> 0.199
            Spec(8, 64, BoxSizes(194, 224), [2]), # 0.449 -> 0.398
            Spec(4, 128, BoxSizes(307, 347), [2]), # 0.599
        ]
    elif model_name == 'B1':
        return [
            Spec(68, 8, BoxSizes(18, 22), [2]), # 0.039
            Spec(34, 16, BoxSizes(37, 48), [2]), # 0.099
            Spec(17, 32, BoxSizes(81, 119), [2]), # 0.238 -> 0.199
            Spec(8, 64, BoxSizes(194, 224), [2]), # 0.449 -> 0.398
            Spec(4, 128, BoxSizes(307, 347), [2]), # 0.599
        ]

    elif model_name == 'B2':
        return [
            Spec(72, 8, BoxSizes(18, 22), [2]),  # 0.039
            Spec(36, 16, BoxSizes(37, 48), [2]),  # 0.099
            Spec(18, 32, BoxSizes(81, 119), [2]),  # 0.238 -> 0.199
            Spec(8, 64, BoxSizes(194, 224), [2]),  # 0.449 -> 0.398
            Spec(4, 128, BoxSizes(307, 347), [2]),  # 0.599
        ]


"""
0529 B2 Input size 544

            Spec(68, 8, BoxSizes(18, 22), [2]), # 0.039
            Spec(34, 16, BoxSizes(37, 48), [2]), # 0.099
            Spec(17, 32, BoxSizes(81, 119), [2]), # 0.238 -> 0.199
            Spec(8, 64, BoxSizes(194, 224), [2]), # 0.449 -> 0.398
            Spec(4, 128, BoxSizes(307, 347), [2]), # 0.599

기존 augmentation 방법으로 학습 
sgd momentum 약 250epoch 학습

  AP 결과
{'aeroplane': 0.8634389240911035,
 'bicycle': 0.8683981914576192,
 'bird': 0.8785208592011551,
 'boat': 0.828494806422587,
 'bottle': 0.6254045336515125,
 'bus': 0.8711601450249413,
 'car': 0.8687860727647553,
 'cat': 0.89812422728918,
 'chair': 0.6886957859100759,
 'cow': 0.8586690603806931,
 'diningtable': 0.7274975275276174,
 'dog': 0.8780796203018911,
 'horse': 0.8785347115372254,
 'motorbike': 0.8796464633922897,
 'person': 0.8523341250203799,
 'pottedplant': 0.663297841135211,
 'sheep': 0.856575101536427,
 'sofa': 0.7836286951668396,
 'train': 0.885383040768544,
 'tvmonitor': 0.8387045052330191}
mAP결과: 0.8246687118906533
               
"""