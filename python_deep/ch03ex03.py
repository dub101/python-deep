import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense


def smooth_curve(points, factor=0.6):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def k_fold_validation(train_data, train_targets, k=4):
    num_val_samples = len(train_data) // k

    num_epochs = 250
    all_scores = []
    all_mae_histories = []

    for i in range(k):
        print(f"Processing fold #{i}")
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                [train_data[:i * num_val_samples],
                train_data[(i + 1) * num_val_samples:]],
                axis=0)
        partial_train_targets = np.concatenate(
                [train_targets[:i * num_val_samples],
                train_targets[(i + 1) * num_val_samples:]],
                axis=0)
        
        model = build_model(partial_train_data)
        history = model.fit(partial_train_data, partial_train_targets,
                validation_data=(val_data, val_targets),
                epochs=num_epochs, batch_size=1, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        mae_history = history.history["mae"]
        all_mae_histories.append(mae_history)

    average_score = np.mean(all_scores)
    print(average_score)
    average_mae_history = [np.mean([x[i] for x in all_mae_histories])
            for i in range(num_epochs)]
    
    return average_mae_history


def build_model(train_data):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


def main():

    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data -= mean
    train_data /= std
    test_data -= mean
    test_data /= std

    smoothed_mae_history = smooth_curve(k_fold_validation(train_data, train_targets))[10:]

    plt.plot(range(1, len(smoothed_mae_history) + 1), smoothed_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()
    


if __name__ == "__main__":
    main()
