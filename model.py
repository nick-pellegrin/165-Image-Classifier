import tensorflow as tf
from tensorflow import keras



"""------------------------------------DATASET------------------------------------"""

def create_ds():

    # Load the dataset with One-Hot Encoding Labels
    train_ds = keras.preprocessing.image_dataset_from_directory(
        "hw4_train\hw4_train",
        labels = "inferred",
        label_mode = "categorical",
        class_names = None,
        color_mode = "grayscale",
        batch_size = 32,
        image_size = (28, 28),
        shuffle = True,
        seed = 17,
        validation_split = .2,
        subset = "training",
    )
    valid_ds = keras.preprocessing.image_dataset_from_directory(
        "hw4_train\hw4_train",
        labels = "inferred",
        label_mode = "categorical",
        class_names = None,
        color_mode = "grayscale",
        batch_size = 32,
        image_size = (28, 28),
        shuffle = True,
        seed = 17,
        validation_split = .2,
        subset = "validation",
    )
    
    return train_ds, valid_ds



"""---------------------------------CREATING MODEL--------------------------------"""

def create_model():
    
    # # Create the VGG-16 Convolutional Neural Network model
    # model_vgg16 = keras.Sequential()

    # model_vgg16.add(keras.layers.Rescaling(1./255, input_shape = (28,28,1)))

    # model_vgg16.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same', input_shape = (28,28,1)))
    # model_vgg16.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.MaxPooling2D((2,2), strides = (2,2)))

    # model_vgg16.add(keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.MaxPooling2D((2,2), strides = (2,2)))

    # model_vgg16.add(keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.MaxPooling2D((2,2), strides = (2,2)))

    # model_vgg16.add(keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    # model_vgg16.add(keras.layers.MaxPooling2D((2,2), strides = (2,2)))

    # # model_vgg16.add(keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    # # model_vgg16.add(keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    # # model_vgg16.add(keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    # # model_vgg16.add(keras.layers.MaxPooling2D((2,2), strides = (2,2)))

    # model_vgg16.add(keras.layers.Flatten())
    # model_vgg16.add(keras.layers.Dense(4096, activation = 'relu'))
    # model_vgg16.add(keras.layers.Dropout(0.25))

    # model_vgg16.add(keras.layers.Dense(4096, activation = 'relu'))
    # model_vgg16.add(keras.layers.Dropout(0.25))

    # model_vgg16.add(keras.layers.Dense(10, activation = 'softmax'))

    # Create the Convolutional Neural Network model
    model = keras.Sequential()

    model.add(keras.layers.Rescaling(1./255, input_shape = (28,28,1)))

    model.add(keras.layers.Conv2D(32, (7,7), activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (5,5), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(128, (3,3), activation = 'relu'))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation = 'relu'))

    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10, activation = 'softmax'))

    
    # Compile the model
    model.compile(
        optimizer = 'adam', 
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy'])
    return model



"""---------------------------------TRAINING MODEL--------------------------------"""

def train_model(model, train_ds, valid_ds):
    # Train the model
    model.fit(train_ds, epochs = 10, validation_data = valid_ds)
    return model



"""-----------------------------------MAIN LOOP-----------------------------------"""

if __name__ == "__main__":
    train_ds, valid_ds = create_ds()
    model = create_model()
    model = train_model(model, train_ds, valid_ds)
    model.save('model')
