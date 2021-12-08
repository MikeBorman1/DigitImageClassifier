# DigitImageClassifier
A Convolutional Neural net in TensorFlow using relu and softmax activation functions on a dataset of written digit images from sklearn

How to run:
To run the program simply execute the python file, recommended in console and not in IDLE

Program gives two options

1. To retrain both both algorithms and save their models(name of model can be changed in
code or it will re-wright the model files)
2. To load models, look at their summary and check accuracy using the test data set
Details of implementation:
The first part of the code loads in the digit data set from sklearn and splits it into test and training
sets.

1. In function trainNSave() the models for the two neural networks are created in the
keras.sequential brackets where the layers are added. The main difference between the
two models is the conv layer followed by a flatten layer.
The models are saved using model.save() which saves the file as a h5 file containing all
the weights of the layers.
For both models the testing data set is also fitted onto the model after it has been trained
to show accuracy, which is printed

2. In the function loadNDisplay() both models are loaded in and the stats are displayed.
They are then both tested against the testing data set to show for accuracy
