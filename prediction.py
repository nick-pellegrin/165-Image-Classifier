import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

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

# def normalize(image):
#     image = tf.cast(image/255. ,tf.float32)
#     return image
# def normalize(image,label):
#     image = tf.cast(image/255. ,tf.float32)
#     return image,label


    
if __name__ == "__main__":

    model = keras.models.load_model('model_2') 

    # test_ds = keras.preprocessing.image_dataset_from_directory(
    #     "hw4_test/hw4_test",
    #     labels = None,
    #     label_mode = None,
    #     class_names = None,
    #     color_mode = "grayscale",
    #     batch_size = 32,
    #     image_size = (256, 256),
    #     shuffle = False,
    #     seed = None,
    #     validation_split = None,
    #     subset = None,
    #     interpolation = "bilinear",
    #     follow_links = False,
    #     crop_to_aspect_ratio = False,
    # )
    #test_ds = test_ds.map(normalize)

    # for i in range(10000):
    #     filename = "hw4_test/hw4_test/{}.png".format(i)
    #     with open(filename, 'rb') as f:
    #         img = tf.io.decode_png(f.read(), channels=3)
    #         img = tf.image.resize(img, [168, 168])
    #         img = tf.cast(img/255. ,tf.float32)
    #         img = tf.expand_dims(img, 0)
    #         if i == 0:
    #             test_ds = img
    #         else:
    #             test_ds = tf.concat([test_ds, img], axis=0)

    # for filename in os.listdir("hw4_test/hw4_test"):
    #     print(filename)
        # img = keras.preprocessing.image.load_img(
        #     filename,
        #     color_mode = "rgb",
        #     target_size=(168,168)
        # )
        # image = tf.cast(img/255. ,tf.float32)
    with open('prediction.txt', 'w') as f:
        for i in range(10000):
            filename = "hw4_test/hw4_test/{}.png".format(i)
            img = keras.preprocessing.image.load_img(
            filename,
            color_mode = "rgb",
            target_size=(168,168)
            )
            #img = keras.preprocessing.image.img_to_array(img, data_format=None, dtype=None)
            img = tf.expand_dims(img, 0)
            prediction = np.argmax(model.predict(img), axis=-1)
        
            f.write(str(prediction[0]))
            f.write('\n')
        

    # filename = "hw4_test/hw4_test/{}.png".format(0)
    # img = keras.preprocessing.image.load_img(
    #     filename,
    #     color_mode = "rgb",
    #     target_size=(168,168)
    # )
    
    
    #plt.imshow(img)
    #image = tf.cast(img/255. ,tf.float32)

    # plt.figure(figsize=(10, 10))
    # for images in test_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.axis("off")
    #plt.show()


    

    #result = model.evaluate(img)


    # predictions = np.argmax(model.predict(test_ds), axis=-1)
    # print(prediction)
    # with open('prediction_2.txt', 'w') as f:
    #      for prediction in predictions:
    #         f.write(str(prediction))
    #         f.write('\n')


    

    
