import tensorflow as tf
import numpy as np
tensor = [0, 1, 2, 3]  # 1-D example
mask = np.array([False, False, False, False])

locations = tf.boolean_mask(tensor, mask)

print(tf.cast(tf.shape(locations)[0], tf.float32))