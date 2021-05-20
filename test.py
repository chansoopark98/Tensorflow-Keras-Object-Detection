import sys

import tensorflow as tf
import numpy as np

epsilon = tf.keras.backend.epsilon()
alpha = 0.25
gamma = 2.0
ce_true = tf.convert_to_tensor([0,0,0,1,0], tf.float32)
ce_pred = tf.convert_to_tensor([0,0,1,0,0], tf.float32)

ce_pred = tf.clip_by_value(ce_pred, epsilon, 1.0 - epsilon)
cross_entropy = -ce_true * tf.math.log(ce_pred)
weight = alpha * ce_true * tf.math.pow((1 - ce_pred), gamma)
loss = weight * cross_entropy
loss = tf.math.reduce_sum(loss)
print(loss)
