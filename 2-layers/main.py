import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


def print_var(var, should_print_content: bool = False):
    if var is not None:
        try:
            print(var.__name__ + ": ", end="")
        except Exception as e:
            print(e)
        print(type(X_train), end="; ")
        print(len(X_train), end="; ")
        try:
            print(X_train[0].shape)
        except Exception as e:
            print(e)
        if should_print_content:
            print("Content: " + var)


if __name__ == "__main__":
    '''Loading input data'''
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # print_var(X_train)
    # print_var(y_train)
    # print_var(X_test)
    # print_var(y_test)
    # plt.matshow(X_train[0])
    # plt.show()
    '''Cleaning data'''
    X_train = X_train / 255
    X_test = X_test / 255
    '''Shaping data'''
    X_train_flattened = X_train.reshape(len(X_train), 28 ** 2)
    X_test_flattened = X_test.reshape(len(X_test), 28 ** 2)
    '''Creating ML model'''
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")
    ])
    '''Specifying model parameters'''
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    '''Running Learning process'''
    model.fit(X_train_flattened, y_train, epochs=5)
    '''Testing model'''
    model.evaluate(X_test_flattened, y_test)
    '''Prediction'''
    y_predicted = model.predict(X_test_flattened)
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

    plt.figure(figsize=(10, 7))
    sn.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()
