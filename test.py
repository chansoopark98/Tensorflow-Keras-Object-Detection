import sys

import tensorflow as tf
import numpy as np
locations = tf.convert_to_tensor([0.875,0.875,0.450763732,0.901527464])

output = tf.concat([locations[:2] - locations[2:] / 2,
                    locations[:2] + locations[2:] / 2], tf.rank(locations) - 1)

print(locations[:2] - locations[2:] / 2)
print(locations[:2] + locations[2:] / 2)
print(output)