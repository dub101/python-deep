import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras import layers


class ConvNet2():

    def __init__(self):
        self.split= 0.70
        self.rows = 32
        self.columns = 32
        self.channels = 3
        self.epochs = 50

        self.convolutional_network = self.build_network()
        self.convolutional_network.compile(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.history_dict = self.train()

        self.plot_loss()
        self.plot_accuracy()

    def obtain_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        id_train = np.arange(x_train.shape[0])
        np.random.shuffle(id_train)
        x_train = x_train[id_train]
        y_train = y_train[id_train]
        num_train_records = int(x_train.shape[0] * self.split)
        return (x_train[:num_train_records], 
                y_train[:num_train_records]), (x_train[num_train_records:], 
                        y_train[num_train_records:]), (x_test, 
                                y_test)

    def build_network(self):

        inputs = keras.Input(shape=(self.rows, self.columns, self.channels))
        residual = inputs

        def residual_block(x, filters, pooling=False):
            residual = x
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.2)(x)
            if pooling:
                x = layers.MaxPooling2D(2, padding="same")(x)
                residual = layers.Conv2D(filters, 4, strides=2, padding="same")(residual)
                residual = layers.BatchNormalization()(residual)
                residual = layers.ReLU()(residual)
                residual = layers.Dropout(0.2)(residual)
            elif filters != residual.shape[-1]:
                residual = layers.Conv2D(filters, 1)(residual)
                residual = layers.BatchNormalization()(residual)
                residual = layers.ReLU()(residual)
                residuall = layers.Dropout(0.2)(residual)
            x = layers.add([x, residual])

            return x

        x = residual_block(inputs, filters=32, pooling=True)
        x = residual_block(x, filters=64, pooling=True)
        x = residual_block(x, filters=128, pooling=False)
        residual = layers.Conv2D(128, 8, strides=4, padding="same")(residual)
        residual = layers.BatchNormalization()(residual)
        residual = layers.ReLU()(residual)
        residual = layers.Dropout(0.2)(residual)
        x = layers.add([x, residual])

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.summary()
        keras.utils.plot_model(model)
        return model

    def train(self):
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = self.obtain_data()
        x_train = x_train.astype("float32") / 255
        x_valid = x_valid.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        history = self.convolutional_network.fit(
            x_train, 
            y_train, 
            epochs=self.epochs,
            batch_size=128,
            validation_data=(x_valid, y_valid)
        )
        self.convolutional_network.evaluate(x_test, y_test)

        return history.history

    def plot_loss(self):
        train_loss_values = self.history_dict["loss"]
        valid_loss_values = self.history_dict["val_loss"]
        epochs = range(1, len(train_loss_values) + 1)
        plt.plot(epochs, train_loss_values, "bo", label="Training loss")
        plt.plot(epochs, valid_loss_values, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        train_acc = self.history_dict["accuracy"]
        valid_acc = self.history_dict["val_accuracy"]
        epochs = range(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, "bo", label="Training acc")
        plt.plot(epochs, valid_acc, "b", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()



if __name__ == "__main__":
    network = ConvNet2()


