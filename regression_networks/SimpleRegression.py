import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from tensorflow.keras import layers


class SimpleRegression():

    def __init__(self):
        self.num_words = 10000
        self.epochs = 50
        self.batch_size = 8
        self.num_splits = 4

        self.regression_model = self.build_network()
        self.regression_model.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=["mae"]
        )
        
        self.all_mae_histories = self.train_network()
        self.regression_model.summary()

        self.plot_mae()

        self.plot_mae_trucated()

        self.evaluate_network()


    def obtain_data(self):
        return boston_housing.load_data()

    def build_network(self):
        model = keras.models.Sequential()

        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1))

        return model

    def train_network(self):
        (train_sample, train_label), (test_sample, test_label) = self.obtain_data()

        mean = train_sample.mean(axis=0)
        std = train_sample.std(axis=0)
        train_sample -= mean
        train_sample /= std

        num_val_samples = len(train_sample) // self.num_splits
        all_mae_histories = []
        for i in range(self.num_splits):
            print(f"Processing fold #{i}")
            val_data = train_sample[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = train_label[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                [train_sample[:i * num_val_samples],
                 train_sample[(i + 1) * num_val_samples:]],
                axis=0)
            partial_train_targets = np.concatenate(
                [train_label[:i * num_val_samples],
                 train_label[(i + 1) * num_val_samples:]],
                axis=0)
            model = self.build_network()
            history = self.regression_model.fit(partial_train_data, partial_train_targets,
                                validation_data=(val_data, val_targets),
                                epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            mae_history = history.history["val_mae"]
            all_mae_histories.append(mae_history)

        return all_mae_histories

    def plot_mae(self):
        average_mae_history = [
            np.mean([
                x[i] for x in self.all_mae_histories
            ]) for i in range(self.epochs)
        ]
        plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
        plt.xlabel("Epochs")
        plt.ylabel("Validation MAE")
        plt.show()

    def plot_mae_trucated(self):
        average_mae_history = [
            np.mean([
                x[i] for x in self.all_mae_histories
            ]) for i in range(self.epochs)
        ]
        truncated_mae_history = average_mae_history[5:]
        plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
        plt.xlabel("Epochs")
        plt.ylabel("Validation MAE")
        plt.show()

    def evaluate_network(self):
        (train_sample, train_label), (test_sample, test_label) = self.obtain_data()
        mean = train_sample.mean(axis=0)
        std = train_sample.std(axis=0)
        train_sample -= mean
        train_sample /= std
        test_sample -= mean
        test_sample /= std
        test_mse_score, test_mae_score = self.regression_model.evaluate(test_sample, test_label)

        print(test_mse_score)
        print(test_mae_score)


if __name__ == "__main__":
    network = SimpleRegression()


