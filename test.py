import tensorflow as tf

shape = [2, 4]

#ious = tf.ones(shape)
ious = tf.random.normal(shape,)

axis1 = tf.math.reduce_max(ious, axis=1)
axis0 = tf.math.reduce_max(ious, axis=0)


print(argmax)