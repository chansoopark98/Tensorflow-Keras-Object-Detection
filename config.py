from utils.priors import *

# Spec(38, 8, BoxSizes(30, 60), [2]),
# Spec(19, 16, BoxSizes(60, 111), [2, 3]),
# Spec(10, 32, BoxSizes(111, 162), [2, 3]),
# Spec(5, 64, BoxSizes(162, 213), [2, 3]),
# Spec(3, 100, BoxSizes(213, 264), [2]),
# Spec(1, 300, BoxSizes(264, 315), [2]),


normalize = [20, 20, 20, -1, -1]
num_priors = [3, 3, 3, 3, 3]

MODEL_INPUT_SIZE = {
    'B0-tiny': 300,
    'B0': 512,
    'B1': 544,
    'B2': 576, # 576
    'B3': 608,
    'B4': 640,
    'B5': 672,
    'B6': 704,
    'B7': 736
}

class TrainHyperParams:
    def __init__(self):
        self.optimizer_name = 'adam'
        self.weight_decay = 0.0005
        self.learning_rate = 0.001
        self.sgd_momentum = 0.9
        self.epochs = 200

    def setOptimizers(self):
        if self.optimizer_name == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.sgd_momentum)

        elif self.optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        elif self.optimizer_name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise print("check optimizers name")


def set_priorBox(model_name):
    if model_name == 'B0-tiny':
        return [
            Spec(38, 8, BoxSizes(30, 60), [2]),
            Spec(19, 16, BoxSizes(60, 111), [2, 3]),
            Spec(10, 32, BoxSizes(111, 162), [2, 3]),
            Spec(5, 64, BoxSizes(162, 213), [2, 3]),
            Spec(3, 100, BoxSizes(213, 264), [2]),
            Spec(1, 300, BoxSizes(264, 315), [2]),
        ]
    elif model_name == 'B0':
        return [
            Spec(64, 8, BoxSizes(15, 31), [2]), # 0.029 | 0.041
            Spec(32, 16, BoxSizes(40, 82), [2]), # 0.078 | 0.111       + 0.049
            Spec(16, 32, BoxSizes(92, 184), [2]), # 0.179 | 0.359      + 0.101
            Spec(8, 64, BoxSizes(194, 228), [2]), # 0.378 | 0.445      + 0.199
            Spec(4, 128, BoxSizes(310, 386), [2]), # 0.605 | 0.753     + 0.227
        ]
    elif model_name == 'B1':
        return [
            Spec(68, 8, BoxSizes(16, 22), [2]),
            Spec(34, 16, BoxSizes(42, 60), [2]),
            Spec(17, 32, BoxSizes(92, 184), [2]),
            Spec(8, 64, BoxSizes(206, 242), [2]),
            Spec(4, 128, BoxSizes(329, 410), [2]),
        ]

    elif model_name == 'B2':
        return [
            Spec(72, 8, BoxSizes(17, 24), [2]),  # 0.039
            Spec(36, 16, BoxSizes(45, 64), [2]),  # 0.099
            Spec(18, 32, BoxSizes(103, 207), [2]),  # 0.238 -> 0.199
            Spec(8, 64, BoxSizes(218, 256), [2]),  # 0.449 -> 0.398
            Spec(4, 128, BoxSizes(348, 434), [2]),  # 0.599
        ]
    elif model_name == 'B3':
        return [
            Spec(76, 8, BoxSizes(18, 25), [2]),  # 0.039
            Spec(38, 16, BoxSizes(47, 67), [2]),  # 0.099
            Spec(19, 32, BoxSizes(109, 218), [2]),  # 0.238 -> 0.199
            Spec(9, 64, BoxSizes(230, 271), [2]),  # 0.449 -> 0.398
            Spec(5, 128, BoxSizes(368, 458), [2]),  # 0.599
        ]
    elif model_name == 'B4':
        return [
            Spec(80, 8, BoxSizes(19, 26), [2]), # 0.029 | 0.041
            Spec(40, 16, BoxSizes(50, 71), [2]), # 0.078 | 0.111       + 0.049
            Spec(20, 32, BoxSizes(115, 230), [2]), # 0.179 | 0.359      + 0.101
            Spec(9, 64, BoxSizes(242, 285), [2]), # 0.378 | 0.445      + 0.199
            Spec(5, 128, BoxSizes(387, 482), [2]), # 0.605 | 0.753     + 0.227
        ]

    elif model_name == 'B5':
        return [
            Spec(84, 8, BoxSizes(19, 28), [2]), # 0.029 | 0.041
            Spec(42, 16, BoxSizes(52, 75), [2]), # 0.078 | 0.111       + 0.049
            Spec(21, 32, BoxSizes(120, 241), [2]), # 0.179 | 0.359      + 0.101
            Spec(10, 64, BoxSizes(254, 299), [2]), # 0.378 | 0.445      + 0.199
            Spec(5, 128, BoxSizes(407, 506), [2]), # 0.605 | 0.753     + 0.227
        ]
    elif model_name == 'B6': # 704
        return [
            Spec(88, 8, BoxSizes(20, 29), [2]), # 0.029 | 0.041
            Spec(44, 16, BoxSizes(55, 78), [2]), # 0.078 | 0.111       + 0.049
            Spec(22, 32, BoxSizes(126, 253), [2]), # 0.179 | 0.359      + 0.101
            Spec(10, 64, BoxSizes(266, 313), [2]), # 0.378 | 0.445      + 0.199
            Spec(5, 128, BoxSizes(426, 530), [2]), # 0.605 | 0.753     + 0.227
        ]
    elif model_name == 'B7': # 736
        return [
            Spec(92, 8, BoxSizes(21, 30), [2]), # 0.029 | 0.041
            Spec(46, 16, BoxSizes(57, 82), [2]), # 0.078 | 0.111       + 0.049
            Spec(23, 32, BoxSizes(131, 263), [2]), # 0.179 | 0.359      + 0.101
            Spec(11, 64, BoxSizes(277, 326), [2]), # 0.378 | 0.445      + 0.199
            Spec(6, 128, BoxSizes(443, 551), [2]), # 0.605 | 0.753     + 0.227
        ]