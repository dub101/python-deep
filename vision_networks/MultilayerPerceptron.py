import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers


class MultilayerPerceptron():

    def __init__(self):
        self.num_records = 50000

        self.multilayer_perceptron = self.build_network()
        self.multilayer_perceptron.compile(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def obtain_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        idx = np.random.randint(0, x_train.shape[0], self.num_records)
        return (x_train[idx], y_train[idx]), (x_test, y_test)

    def build_network(self):
        model = keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        return model

    def train(self):
        (x_train, y_train), (x_test, y_test) = self.obtain_data()
        x_train= x_train.reshape((self.num_records, 28*28))
        x_train = x_train.astype("float32") / 255
        x_test = x_test.reshape(10000, 28*28)
        x_test = x_test.astype("float32") / 255

        self.multilayer_perceptron.fit(x_train, y_train, epochs=50, batch_size=128)
        self.multilayer_perceptron.evaluate(x_test, y_test)



