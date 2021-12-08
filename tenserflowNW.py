from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np



"""Edit the variable below to chnage the size of test data"""
testDataPercent = 0.26
digits = load_digits()
print("Number classes:10\nNumber of entries:1797\nNumber of entries per class ~180\nFeature values: 0-9 int val")
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, test_size=testDataPercent, random_state=0)
#Flattening data
X_train = X_train/16.0
X_train = X_train/16.0
#reforming the data to allow it to pass into the conv layer
X2_train = X_train.reshape(X_train.shape[0],8,8,1)
X2_test = X_test.reshape(X_test.shape[0],8,8,1)    

def trainNSave():


    #Create model
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
        ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #train model
    model.fit(X_train, y_train, epochs=10)
    #test model on test data
    test_loss, test_acc  =model.evaluate(X_test,y_test)

    print("Test Acc:", test_acc)

    model.save('model.h5')

    #creating conv layer model
    model2 = keras.Sequential([
        keras.layers.Conv2D(8, 3, input_shape=(8, 8,1)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
        ])
    model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model2.fit(X2_train, y_train, epochs=10)

    test_loss, test_acc  =model2.evaluate(X2_test,y_test)

    print("Test Acc:", test_acc)

    model2.save('convModel.h5')

def loadNDisplay():
    model = keras.models.load_model('model.h5')
    model.summary()
    print("Testing model on test data")
    print()
    test_loss, test_acc  =model.evaluate(X_test,y_test)

    print("Test Acc:", test_acc)


    print()
    model2 = keras.models.load_model('convModel.h5')
    model2.summary()
    print("Testing model on test data")
    print()
    test_loss, test_acc  =model2.evaluate(X2_test,y_test)

    print("Test Acc:", test_acc)


x = input("Press 1 to train and save both models(takes 5 seconds)) \nPress 2 to view saved models and accuracy")
if x == 1:
    trainNSave()
else:
    loadNDisplay()
    


