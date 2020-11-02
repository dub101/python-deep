from keras.datasets import mnist
from keras import models, layers


def main():

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(f"Train images shape:\t{train_images.shape}")
    print(f"Train labels shape:\t{train_labels.shape}")
    print(f"Test images shape:\t{test_images.shape}")
    print(f"Test labels shape:\t{test_labels.shape}")

    network = models.Sequential()
    network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation="softmax"))


if __name__ == "__main__":
    main()

