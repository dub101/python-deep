import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers


class MultilayerPerceptron():

    def __init__(self):
        self.batch_size = 32
        self.epochs = 10
        self.max_tokens = 20000
        self.output_dim = 256
        self.max_length = 600
        self.hidden_dim = 16
        

        self.multilayer_perceptron = self.build_network()
        self.multilayer_perceptron.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.train()

    def obtain_data(self):
        train_ds = keras.utils.text_dataset_from_directory(
            "data/aclImdb/train", batch_size=self.batch_size
        )
        val_ds = keras.utils.text_dataset_from_directory(
            "data/aclImdb/val", batch_size=self.batch_size
        )
        test_ds = keras.utils.text_dataset_from_directory(
            "data/aclImdb/test", batch_size=self.batch_size
        )

        return (train_ds, val_ds, test_ds)

    def build_network(self):

        inputs = keras.Input(shape=(None,), dtype="int64")
        embedded = layers.Embedding(input_dim=self.max_tokens, output_dim=self.output_dim)(inputs)
        x = layers.Bidirectional(layers.LSTM(32))(embedded)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
    
        model = keras.Model(inputs, outputs)
        
        model.summary()
        return model

    def train(self):
        (train_ds, val_ds, test_ds) = self.obtain_data()

        text_vectorization = layers.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode="int",
            output_sequence_length=self.max_length
        )
        text_only_train_ds = train_ds.map(lambda x, y: x)
        text_vectorization.adapt(text_only_train_ds)

        binary_1gram_train_ds = train_ds.map(
            lambda x, y: (text_vectorization(x), y),
            num_parallel_calls=4
        )
        binary_1gram_val_ds = val_ds.map(
            lambda x, y: (text_vectorization(x), y),
            num_parallel_calls=4
        )
        binary_1gram_test_ds = test_ds.map(
            lambda x, y: (text_vectorization(x), y),
            num_parallel_calls=4
        )

        self.multilayer_perceptron.fit(
            binary_1gram_train_ds,
            epochs = self.epochs,
            validation_data = binary_1gram_val_ds,
        )
        self.multilayer_perceptron.evaluate(binary_1gram_test_ds)


if __name__ == "__main__":
    network = MultilayerPerceptron()


