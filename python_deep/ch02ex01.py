from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical


def main():

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(f"Train images shape:\t{train_images.shape}")
    print(f"Train labels shape:\t{train_labels.shape}")
    print(f"Test images shape:\t{test_images.shape}")
    print(f"Test labels shape:\t{test_labels.shape}")

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    train_labels = to_categorical(train_labels)
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    test_labels = to_categorical(test_labels)

    network = models.Sequential()
    network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation="softmax"))
    network.compile(optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

    network.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(f"Test acc:\t{test_acc}")


if __name__ == "__main__":
    main()

