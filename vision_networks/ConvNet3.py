import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory


class ConvNet2():

    def __init__(self):
        self.split= 0.70
        self.rows = 180
        self.columns = 180
        self.channels = 3
        self.epochs = 50

        self.convolutional_network = self.build_network()
        self.convolutional_network.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.history_dict = self.train()

        self.plot_loss()
        self.plot_accuracy()

    def obtain_data(self):
        train_dataset = image_dataset_from_directory(
            "data/cats_vs_dogs_small/train",
            image_size=(180, 180),
            batch_size=32)
        validation_dataset = image_dataset_from_directory(
            "data/cats_vs_dogs_small/validation",
            image_size=(180, 180),
            batch_size=32)
        test_dataset = image_dataset_from_directory(
            "data/cats_vs_dogs_small/test",
            image_size=(180, 180),
            batch_size=32)
        return (train_dataset, validation_dataset, test_dataset)

    def build_network(self):
        model = keras.Sequential([
            layers.Input(shape=(self.rows, self.columns, self.channels)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
            layers.Rescaling(1/255),
            layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu"),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid")
        ])
        model.summary()
        return model

    def train(self):
        (train_dataset, validation_dataset, test_dataset) = self.obtain_data()

        history = self.convolutional_network.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=validation_dataset
        )
        self.convolutional_network.evaluate(test_dataset)

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


