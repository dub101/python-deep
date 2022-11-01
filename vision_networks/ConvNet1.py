import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers


class ConvNet1():

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
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
            layers.Flatten(),
            layers.Dense(10, activation="softmax")
        ])
        model.summary()
        return model

    def train(self):
        (x_train, y_train), (x_test, y_test) = self.obtain_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        self.multilayer_perceptron.fit(x_train, y_train, epochs=50, batch_size=128)
        self.multilayer_perceptron.evaluate(x_test, y_test)


if __name__ == "__main__":
    network = ConvNet1()
    network.train()


