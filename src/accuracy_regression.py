import tensorflow as tf
import numpy as np

# Method to compute accuracy for the regression task
def accuracy_regression_helper(y_expected, y_predicted):
    y_expected = np.copy(y_expected)

    y_predicted = np.copy(y_predicted)
    y_predicted = np.reshape(y_predicted, (y_predicted.shape[0]))
    y_predicted = np.around(y_predicted)

    # clip the values that are either too low or too high
    for i in range(len(y_predicted)):
        if y_predicted[i] <= 1:
            y_predicted[i] = 1
        elif y_predicted[i] >= 5:
            y_predicted[i] = 5

    assert len(y_predicted) == len(y_expected)

    # compute batch accuracy
    accuracy = sum([1 for i in range(len(y_predicted)) if y_predicted[i] == y_expected[i]])
    return accuracy / len(y_predicted)


def accuracy_regression(y_true, y_pred):
    return tf.py_function(accuracy_regression_helper, (y_true, y_pred), tf.double)