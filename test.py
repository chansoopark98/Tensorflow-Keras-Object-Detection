import tensorflow as tf
import numpy as np

tensor = [[1,2,3,4,tf.convert_to_tensor(np.NINF),6,7,8,9],[11,12,13,14,tf.convert_to_tensor(np.NINF),16,17,18,19]]
tensor = tf.convert_to_tensor(tensor)
num_neg = [[2],[3]]
print(tensor)
indexes = tf.argsort(tensor, axis=1, direction='DESCENDING')  # N, N 내림차순 정렬
print(indexes)
orders = tf.argsort(indexes, axis=1)  # N, N 오름차순 정렬
print(orders)
neg_mask = tf.cast(orders, tf.float32) < num_neg # N, N
print(neg_mask)