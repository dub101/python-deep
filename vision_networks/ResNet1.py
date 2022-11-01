import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers


class ResNet1():

    def __init__(self):
        self.num_records = 50000

        self.residual_network = self.build_network()
        self.residual_network.compile(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def obtain_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        idx = np.random.randint(0, x_train.shape[0], self.num_records)
        return (x_train[idx], y_train[idx]), (x_test, y_test)

    def build_network(self):

        inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Rescaling(1/255)(inputs)

        def residual_block(x, filters, pooling=False):
            residual = x
            x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
            x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
            if pooling:
                x = layers.MaxPooling2D(2, padding="same")(x)
                residual = layers.Conv2D(filters, 1, strides=2)(residual)
            elif filters != residual.shape[-1]:
                residual = layers.Conv2D(filters, 1)(residual)
            x = layers.add([x, residual])

            return x

        x = residual_block(x, filters=32, pooling=True)
        x = residual_block(x, filters=64, pooling=True)
        x = residual_block(x, filters=128, pooling=False)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.summary()
        keras.utils.plot_model(model)
        return model

    def train(self):
        (x_train, y_train), (x_test, y_test) = self.obtain_data()

        self.residual_network.fit(
            x_train, 
            y_train, 
            epochs=50, 
            batch_size=128,
            validation_data=(x_test, y_test)
        )
        self.residual_network.evaluate(x_test, y_test)


if __name__ == "__main__":
    network = ResNet1()
    network.train()


