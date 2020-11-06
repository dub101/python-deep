from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


def main():

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
            "../data/cat_dog_small/train",
            target_size=(150, 150),
            batch_size=20,
            class_mode="binary")

    validation_generator = validation_datagen.flow_from_directory(
            "../data/cat_dog_small/validation",
            target_size=(150, 150),
            batch_size=20,
            class_mode="binary")
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150,  150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(loss="binary_crossentropy",
            optimizer=optimizers.RMSprop(lr=0.0001),
            metrics=["acc"])

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50)

    model.save("cat_dogs_small_1.h5")
    
    

if __name__ == "__main__":
    main()
