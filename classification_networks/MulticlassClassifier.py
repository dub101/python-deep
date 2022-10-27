import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow import keras
from tensorflow.keras import layers


class MulticlassClassifier():

    def __init__(self):
        self.num_words = 10000
        self.epochs = 20
        self.batch_size = 512
        self.num_classes = 46
        self.max_length = 200

        self.binary_classifier = self.build_network()
        self.binary_classifier.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.history_dict = self.train_network()
        self.binary_classifier.summary()

        self.plot_loss()

        self.plot_accuracy()

        self.evaluate_network()


    def obtain_data(self):
        return reuters.load_data(num_words=self.num_words)

    def vectorize_data(self, sequences):
        results = np.zeros((len(sequences), self.num_words))
        for i, sequence in enumerate(sequences):
            for j in sequence:
                results[i, j] = 1
        return results

    def one_hot_encode(self, labels):
        results = np.zeros((len(labels), self.num_classes))
        for i, label in enumerate(labels):
            results[i, label] = 1
        return results


    def build_network(self):
        model = keras.models.Sequential()

        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(self.num_classes, activation="softmax"))

        return model

    def train_network(self):
        (train_sample, train_label), (test_sample, test_label) = self.obtain_data()

        x_train = self.vectorize_data(train_sample[:5000])
        x_valid = self.vectorize_data(train_sample[5000:])
        print(x_train.shape[0])
        print(x_valid.shape[0])

        y_train = self.one_hot_encode(train_label[:5000]).astype("float32")
        y_valid = self.one_hot_encode(train_label[5000:]).astype("float32")

        history = self.binary_classifier.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(x_valid, y_valid)
        )

        history_dict = history.history

        return history_dict

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

    def evaluate_network(self):
        (train_sample, train_label), (test_sample, test_label) = self.obtain_data()
        x_test = self.vectorize_data(test_sample)
        y_test = self.one_hot_encode(test_label).astype("float32")
        results = self.binary_classifier.evaluate(x_test, y_test)
        print(results)


if __name__ == "__main__":
    network = MulticlassClassifier()

