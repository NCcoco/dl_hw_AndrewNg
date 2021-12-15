import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)
print(tf.keras.__version__)

# 使用 tf 2.0 版本实现。 

y_hat = tf.constant(39.0, name='y_hat', dtype=float)
y = tf.constant(36, name='y', dtype=float)
print(y_hat)

loss = (y_hat - y)**2

print(loss)

a = tf.constant(2)
b = tf.constant(10)
print(tf.multiply(a, b))


print(tf.compat.v1.zeros_initializer())

tf.set_random_seed(1)
