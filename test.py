import sys

import tensorflow as tf
import numpy as np

epsilon = tf.keras.backend.epsilon()

tensor = tf.convert_to_tensor([-0.1, 1, 2, 3, -0.88], tf.float32)

tensor = tf.clip_by_value(tensor, clip_value_min=-1000, clip_value_max=1-epsilon)
print(tensor)
