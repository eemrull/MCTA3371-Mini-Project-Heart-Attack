import numpy as np


def sigmoid(inputs: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-inputs))


def ReLU(inputs: np.ndarray) -> np.ndarray:
    return np.maximum(0, inputs)


def softmax(inputs: np.ndarray) -> np.ndarray:
    exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
