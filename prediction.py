import tensorflow as tf
from tensorflow import keras
import numpy as np
# import matplotlib.pyplot as plt
# import os

"""
Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.
"""

def normalize(image):
    image = tf.cast(image/255. ,tf.float32)
    return image

if __name__ == "__main__":

    # i = 0
    # for file in os.listdir('./hw4_test/hw4_test'):
    #     print(i)
    #     i += 1

    test_ds = keras.preprocessing.image_dataset_from_directory(
        "hw4_test/hw4_test",
        labels = None,
        label_mode = None,
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
    test_ds = test_ds.map(normalize)


    model = keras.models.load_model('model')

    predictions = np.argmax(model.predict(test_ds), axis=-1)

    
