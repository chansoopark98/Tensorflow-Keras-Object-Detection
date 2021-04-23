import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D

tensor = tf.ones([1, 704, 704,3])

tensor = Conv2D(3, 3, (2, 2), padding='same')(tensor) # 352
tensor = Conv2D(3, 3, (2, 2), padding='same')(tensor) # 176
tensor = Conv2D(3, 3, (2, 2), padding='same')(tensor) # 88

tensor = Conv2D(3, 3, (2, 2), padding='same')(tensor) # 44
tensor = Conv2D(3, 3, (2, 2), padding='same')(tensor) # 22
tensor = Conv2D(3, 3, (2, 2), padding='same')(tensor) # 11

pool1 = MaxPooling2D(pool_size=3, strides=2, padding='valid')(tensor) # 4x4
pool2 = MaxPooling2D(pool_size=3, strides=2, padding='valid')(pool1) # 4x4
print(pool1.shape)
print(pool2.shape)