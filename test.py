import tensorflow as tf


None_value = tf.constant([], dtype=tf.float32)

a = tf.where(tf.equal(tf.size(None_value),0),1,2)\

print(a)
bbox = tf.constant([[1,2,3,4],[5,6,7,8]], dtype=tf.float32)
bbox = tf.stack([bbox[:, 1], bbox[:, 0], bbox[:, 3], bbox[:, 2]], axis=1)
print(bbox )