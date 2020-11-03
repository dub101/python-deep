import numpy as np
import matplotlib.pyplot as plt
from  keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")

    model = Sequential()
    model.add(Dense(16, activation="relu", input_shape=(10000,)))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["acc"])

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    history = model.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))

    history_dict = history.history
    print(history_dict.keys())

    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    acc_values = history_dict["acc"]
    val_acc_values = history_dict["val_acc"]
    epochs = range(1, 21)

    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, acc_values, "bo", label="Training acc")
    plt.plot(epochs, val_acc_values, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    print(model.predict(x_test))
    

    


if __name__ == "__main__":
    main()

