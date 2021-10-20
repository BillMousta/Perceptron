import numpy as np

class Perceptron:

    """
    Perceptron algorithm
    """
    def perceptron(self, data_train, labels_train, data_test, learning_rate, epochs):
        predictions = []
        weights, bias = self.train_weights(data_train, labels_train, learning_rate, epochs)
        for x in data_test:
            predictions.append(self.predict_class(x, weights, bias))

        return predictions

    """
    This Function try to predict in which class the data x belongs.
    y = w^T x + bias
    """
    @staticmethod
    def predict_class(x, weights, bias):
        y = np.dot(weights.transpose(), x) + bias
        if y >= 0:
            return 1
        else:
            return 0

    """
    Estimate Perceptron weights using stochastic gradient descent
    Update weights according to the form w(t+1)= w(t) + learning_rate * (expected(t) - predicted(t)) * x(t)
    labels are the actual classes for each row in data x
    0 < learning rate <= 1
    epochs is the number of epochs
    """
    def train_weights(self, data, labels, learning_rate, epochs, bias=0):
        weights = np.zeros(len(data[0]))

        for epoch in range(epochs):
            sum_error = 0
            for i, x in enumerate(data):
                prediction = self.predict_class(x, weights, bias)
                error = labels[i] - prediction
                sum_error += error**2
                # update bias
                bias = bias + learning_rate*error
                weights = weights + learning_rate * error * x

        return weights, bias
