import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(np.round(classifications[0], 4))
print(test_labels[0])
plt.imshow(test_images[0])
plt.show()

