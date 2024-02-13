from fashion_mnist_data import train_images, train_labels, test_images, test_labels, fashion_mnist
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

def visualize_image(index):
    np.set_printoptions(linewidth=320)

    print(f'LABEL: {train_labels[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n {train_images[index]}')

    plt.imshow(train_images[index], cmap="Greys")
    plt.show()

class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\n Accuracy of 90% reached")
            self.model.stop_training = True

callbacks = MyCallback()



model = tf.keras.Sequential( [
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


train_images = train_images / 255.0
test_images = test_images / 255.0

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10, callbacks = [callbacks])

model.evaluate(test_images, test_labels)
