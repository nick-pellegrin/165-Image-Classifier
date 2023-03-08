import tensorflow as tf
from tensorflow import keras



"""---------------------------------PREPROCESSING---------------------------------"""

def normalize(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

def preprocessing():
    # Load the dataset with One-Hot Encoding Labels
    train_ds = keras.preprocessing.image_dataset_from_directory(
        "hw4_train\hw4_train",
        labels = "inferred",
        label_mode = "categorical",
        class_names = None,
        color_mode = "grayscale",
        batch_size = 32,
        image_size = (256, 256),
        shuffle = True,
        seed = None,
        validation_split = None,
        subset = None,
        interpolation = "bilinear",
        follow_links = False,
        crop_to_aspect_ratio = False,
    )
    train_ds = train_ds.map(normalize)
    return train_ds



"""---------------------------------CREATING MODEL--------------------------------"""

def create_model():
    # Create the Convolutional Neural Network model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (256,256,1)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(
        optimizer = 'adam', 
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy'])
    return model



"""---------------------------------TRAINING MODEL--------------------------------"""

def train_model(model, ds):
    # Train the model
    model.fit(ds, epochs = 10)
    return model



"""-----------------------------------MAIN LOOP-----------------------------------"""

if __name__ == "__main__":
    ds = preprocessing()
    model = create_model()
    model = train_model(model, ds)
    model.save('model')