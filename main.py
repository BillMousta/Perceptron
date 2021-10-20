import numpy as np
import Perceptron
from csv import reader
from random import seed
from random import randrange

# Load data from csv file
def load_data(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


"""
Separate sonar data into to numpy array one with data values (float) and the other with labels (which class 1 or 0)
"""
def separate_data_from_label(dataset):
    labels = []
    class1 = dataset[0][-1] # for separate classes
    for i, row in enumerate(dataset):
        if row[-1] == class1:
            labels.append(0)
            dataset[i].remove(row[-1])
        else:
            labels.append(1)
            dataset[i].remove(row[-1])

        for j in range(len(row)):
            dataset[i][j] = float(row[j])

    return np.array(labels), np.array(dataset)


# Split a dataset into k folds
def cross_validation_split(dataset, labels, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    labels_split = list()
    labels_copy = list(labels)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold_data = list()
        fold_label = list()
        while len(fold_data) < fold_size:
            index = randrange(len(dataset_copy))
            fold_data.append(dataset_copy.pop(index))
            fold_label.append(labels_copy.pop(index))

        dataset_split.append(fold_data)
        labels_split.append(fold_label)


    return dataset_split, labels_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


"""
Using cross validation split to split data into train and test 
"""
def evaluate_algorithm(dataset, labels, algorithm, n_folds, learning_rate, n_epoch):
    fold_data, fold_label = cross_validation_split(dataset, labels,  n_folds)
    scores = list()
    for i, fold in enumerate(fold_data):

        train_set_data = list(fold_data)
        train_set_data.pop(i)

        train_set_label = list(fold_label)
        train_set_label.remove(fold_label[i])

        # merge data into 1 list
        train_set_data = sum(train_set_data, [])
        train_set_label = sum(train_set_label, [])

        test_set_data = list()
        test_set_label = list(fold_label[i])
        for row in fold:
            test_set_data.append(row)

        predicted = algorithm.perceptron(np.array(train_set_data), np.array(train_set_label), np.array(test_set_data), learning_rate, n_epoch)
        accuracy = accuracy_metric(test_set_label, predicted)
        scores.append(accuracy)

    return scores


if __name__ == '__main__':
    # Sonar data
    # Banknote authentication data
    seed(1)
    # Choose data sonar or bank
    # filename = 'sonar.all-data'
    filename = 'data_banknote_authentication.txt'
    dataset = load_data(filename)
    labels, dataset = separate_data_from_label(dataset)

    # Change these parameters to improve or to impair the score
    n_folds = 4
    learning_rate = 0.05
    n_epoch = 400

    # for testing data using cross validation algorithm
    p = Perceptron.Perceptron()
    scores = evaluate_algorithm(dataset, labels, p, n_folds, learning_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    print(2)

