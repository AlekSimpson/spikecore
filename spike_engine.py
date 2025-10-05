import tensorflow as tf
import numpy as np
import time

# def tf_add(vector1: tf.Variable, vector2: tf.Tensor):
#     return vector1.assign_add(vector2)

x = 100000000

# Your original vector addition code
np_array = np.arange(x, dtype=np.float32)
vec1 = tf.Variable(tf.convert_to_tensor(np_array))
vec2 = tf.ones(x)
buffer = tf.zeros(x)

start = time.perf_counter()
vec1.assign_add(vec2)
end = time.perf_counter()
# print(f'tf vec1 is: {vec1.numpy()}')
print(f'metal operation took: {end - start} seconds')

np_array = np.arange(x, dtype=np.float32)
vec1 = tf.constant(tf.convert_to_tensor(np_array))
vec2 = tf.ones(x)
buffer = tf.zeros(x)

start = time.perf_counter()
buffer = vec1 + vec2
end = time.perf_counter()
print(f'metal buff version: {end - start} seconds')



