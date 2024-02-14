import numpy as np
from typing import Iterable


def load_data(filepath: str = 'data.csv'):
    with open(filepath, 'r') as f:
        f.readline()
        X, y = [], []
        while line := f.readline():
            *X_data, y_data = map(float, line.strip().split(','))
            X.append(X_data)
            y.append(y_data)

    ratio = 0.1
    length = len(X)
    split_slice = int(length * ratio)
    X_test, X_train = np.array(X[:split_slice]), np.array(X[split_slice:])
    y_test, y_train = np.array(y[:split_slice]), np.array(y[split_slice:])

    y_test = y_test.astype(np.int64)
    y_train = y_train.astype(np.int64)
    new_y_test = np.zeros((y_test.size, y_test.max() + 1))
    new_y_test[np.arange(y_test.size), y_test] = 1
    new_y_train = np.zeros((y_train.size, y_train.max() + 1))
    new_y_train[np.arange(y_train.size), y_train] = 1
    y_test = new_y_test
    y_train = new_y_train

    return np.array(X_test), np.array(X_train), np.array(y_test), np.array(y_train)


def mse(error_list: float):
    return np.sum(np.power(error_list, 2))/len(error_list)


def divide_chunks(items: Iterable, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
