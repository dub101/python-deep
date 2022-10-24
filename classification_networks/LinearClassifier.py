import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers


class LinearClassifier():

    def __init__(self):
        self.num_samples_per_class = 1000
        self.input_dim = 2
        self.output_dim = 1
        self.learning_rate = 0.1

        self.W = tf.Variable(initial_value=tf.random.uniform(shape=(self.input_dim, self.output_dim)))
        self.b = tf.Variable(initial_value=tf.zeros(shape=(self.output_dim,)))

        (self.inputs, self.targets) = self.obtain_data()

        self.loss = self.train()
        print(f"Loss: {self.loss}")

        self.inference()

    def obtain_data(self):
        negative_samples = np.random.multivariate_normal(
            mean=[0,2],
            cov=[[1, 0.5],[0.5, 1]],
            size=self.num_samples_per_class
        )
        positive_samples = np.random.multivariate_normal(
            mean=[2,0],
            cov=[[1, 0.5],[0.5, 1]],
            size=self.num_samples_per_class
        )

        inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
        targets = np.vstack(
            (np.zeros((self.num_samples_per_class, 1), dtype="float32"),
            np.ones((self.num_samples_per_class, 1), dtype="float32")
        ))
        
        return (inputs, targets)

    def network(self, inputs):
        return tf.matmul(inputs, self.W) + self.b

    def square_loss(self, targets, predictions):
        per_sample_losses = tf.square(targets - predictions)
        return tf.reduce_mean(per_sample_losses)


    def training_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self.network(inputs)
            loss = self.square_loss(targets, predictions)
        grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [self.W, self.b])
        self.W.assign_sub(grad_loss_wrt_W * self.learning_rate)
        self.b.assign_sub(grad_loss_wrt_b * self.learning_rate)
        return loss


    def train(self):
        for step in range(40):
            loss = self.training_step(self.inputs, self.targets)
            print(f"Loss at step {step}: {loss:.4f}")

    def inference(self):
        predictions = self.network(self.inputs)
        plt.scatter(self.inputs[:, 0], self.inputs[:, 1], c=predictions[:, 0] > 0.5)
        plt.show()
        x = np.linspace(-1, 4, 100)
        y = - self.W[0] /  self.W[1] * x + (0.5 - self.b) / self.W[1]
        plt.plot(x, y, "-r")
        plt.scatter(self.inputs[:, 0], self.inputs[:, 1], c=predictions[:, 0] > 0.5)
        plt.show()


if __name__ == "__main__":
    network = LinearClassifier()


