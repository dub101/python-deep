import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers


class MultilayerPerceptron():

    def __init__(self):
        self.train_split = 0.5
        self.valid_split = 0.25
        self.sampling_rate = 6
        self.sequence_length = 120
        self.delay = self.sampling_rate * (self.sequence_length + 24 - 1)
        self.batch_size = 128
        self.epochs = 10
        

        self.multilayer_perceptron = self.build_network()
        self.multilayer_perceptron.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=["mae"]
        )

        self.train()

    def obtain_raw_data(self):
        with open(os.path.join("data", "climate", "jena_climate_2009_2016.csv")) as f:
            data = f.read()

        lines = data.split("\n")
        header = lines[0].split(",")
        lines = lines[1:]

        temperature = np.zeros((len(lines),))
        raw_data = np.zeros((len(lines), len(header) -1))
        
        for i, line in enumerate(lines):
            values = [float(x) for x in line.split(",")[1:]]
            temperature[i] = values[1]
            raw_data[i, :] = values[:]

        return (header, temperature, raw_data)

    def obtain_process_data(self):
        (header, temperature, raw_data) = self.obtain_raw_data()

        num_train_samples = int(self.train_split * len(raw_data))

        mean = raw_data[:num_train_samples].mean(axis=0)
        std = raw_data[:num_train_samples].std(axis=0)
        raw_data -= mean
        raw_data /=std
        
        return (header, temperature, raw_data) 

    def build_network(self):
        (header, temperature, raw_data) = self.obtain_process_data()

        inputs = keras.Input(shape=(self.sequence_length, raw_data.shape[-1]))
        x = layers.Conv1D(8, 24, activation="relu")(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(8, 12, activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(8, 6, activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        
        model.summary()
        return model

    def train(self):
        (header, temperature, raw_data) = self.obtain_process_data()

        train_dataset = keras.utils.timeseries_dataset_from_array(
            raw_data[:-self.delay],
            targets=temperature[self.delay:],
            sampling_rate=self.sampling_rate,
            sequence_length=self.sequence_length,
            shuffle=True,
            batch_size=self.batch_size,
            start_index=0,
            end_index=int(self.train_split * len(raw_data))
        )
        valid_dataset = keras.utils.timeseries_dataset_from_array(
            raw_data[:-self.delay],
            targets=temperature[self.delay:],
            sampling_rate=self.sampling_rate,
            sequence_length=self.sequence_length,
            shuffle=True,
            batch_size=self.batch_size,
            start_index=int(self.train_split * len(raw_data)),
            end_index=int(self.train_split * len(raw_data)) + int(self.valid_split * len(raw_data))
        )
        test_dataset = keras.utils.timeseries_dataset_from_array(
            raw_data[:-self.delay],
            targets=temperature[self.delay:],
            sampling_rate=self.sampling_rate,
            sequence_length=self.sequence_length,
            shuffle=True,
            batch_size=self.batch_size,
            start_index=int(self.train_split * len(raw_data)) + int(self.valid_split * len(raw_data))
        )

        self.multilayer_perceptron.fit(
            train_dataset,
            epochs = self.epochs,
            validation_data = valid_dataset,
        )
        self.multilayer_perceptron.evaluate(test_dataset)


if __name__ == "__main__":
    network = MultilayerPerceptron()


