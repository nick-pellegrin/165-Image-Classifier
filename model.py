import tensorflow as tf
from tensorflow import keras
#import keras_tuner as kt



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

    # # Define the hyperparameters
    # conv_filters = hp.Int('filters', min_value = 28, max_value = 512, step = 1)
    # dense_units = hp.Int('units', min_value = 28, max_value = 512, step = 1)
    # dropout_rate = hp.Float('dropout', min_value = 0.0, max_value = 0.5, step = 0.05)
    # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    # # Create the Convolutional Neural Network model
    # model = keras.Sequential()

    # model.add(keras.layers.Rescaling(1./255, input_shape = (28,28,1)))

    # model.add(keras.layers.Conv2D(conv_filters, (3,3), activation = 'relu', padding = 'same'))
    # if hp.Boolean("batchnormalization1"): model.add(keras.layers.BatchNormalization())
    # if hp.Boolean("maxpooling1"):         model.add(keras.layers.MaxPooling2D((2,2)))
    # if hp.Boolean("dropout1"):            model.add(keras.layers.Dropout(dropout_rate))

    # model.add(keras.layers.Conv2D(conv_filters, (3,3), activation = 'relu', padding = 'same'))
    # if hp.Boolean("batchnormalization2"): model.add(keras.layers.BatchNormalization())
    # if hp.Boolean("maxpooling2"):         model.add(keras.layers.MaxPooling2D((2,2)))
    # if hp.Boolean("dropout2"):            model.add(keras.layers.Dropout(dropout_rate))

    # # model.add(keras.layers.Conv2D(conv_filters, (3,3), activation = 'relu'))
    # # model.add(keras.layers.BatchNormalization())
    # # model.add(keras.layers.Dropout(0.4))

    # if hp.Boolean("flatten"):             model.add(keras.layers.Flatten())
    # if hp.Boolean("dense"):               model.add(keras.layers.Dense(dense_units, activation = 'relu'))

    # if hp.Boolean("dropout3"):            model.add(keras.layers.Dropout(dropout_rate))
    # model.add(keras.layers.Dense(10, activation = 'softmax'))

    
    # # Compile the model
    # model.compile(
    #     optimizer = keras.optimizers.Adam(learning_rate=learning_rate), 
    #     loss = 'categorical_crossentropy', 
    #     metrics = ['accuracy'])
    # return model

    model = keras.Sequential()
    model.add(keras.layers.Rescaling(1./255, input_shape = (28,28,1)))

    model.add(keras.layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D((2,2)))
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(512, activation = 'relu'))
    model.add(keras.layers.Dense(128, activation = 'relu'))
    model.add(keras.layers.Dense(10, activation = 'softmax'))

    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model



"""---------------------------------TRAINING MODEL--------------------------------"""

def train_model(model, train_ds, valid_ds):

    # tuner = kt.BayesianOptimization(
    #     create_model,
    #     objective = "val_accuracy",
    #     max_trials = 10,
    #     executions_per_trial = 5,
    #     overwrite = True,
    #     directory = "tuner_results",
    #     project_directory = "results_1",
    # )

    # tuner.search(train_ds, epochs = 25, validation_data = valid_ds)
    # best_model = tuner.get_best_models(num_models=1)
    # best_model.summary()
    # return best_model


    # Train the model
    model.fit(train_ds, epochs = 25, validation_data = valid_ds)
    return model



"""-----------------------------------MAIN LOOP-----------------------------------"""

if __name__ == "__main__":
    train_ds, valid_ds = create_ds()
    # model = create_model(kt.HyperParameters())
    model = create_model()
    model = train_model(model, train_ds, valid_ds)
    model.save('model')
