# 165-Image-Classifier
This project uses the F-MNIST dataset to train a image recognition model. Similar to recognizing handwritten digits as in the MNIST dataset, the F-MNIST dataset uses pictures of different articles of clothing (such as pants, shirt, etc).  
![](https://github.com/nick-pellegrin/165-Image-Classifier/blob/main/f-mnist.png)
* Model.py was used to create and train the model.  
* Prediction.py was used to load the model and create the predictions file for submission and grading on kaggle.  

The best model (made from scratch through trail and error and some hyperparameter tuning) acheived 92.18% accuracy.  Some methods of improving this model could be to create more models and utilize an aggregate voting method between the models with the validation accuracy for each model acting as the weight for that models prediction in the vote.
