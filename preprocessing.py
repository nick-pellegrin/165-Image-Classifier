import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


if __name__ == "__main__":


    # Load the dataset with One-Hot Encoding Labels
    train_ds = keras.preprocessing.image_dataset_from_directory(
        "hw4_train\hw4_train",
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="grayscale",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(labels[i].numpy().argmax())
            plt.axis("off")
    plt.show()

