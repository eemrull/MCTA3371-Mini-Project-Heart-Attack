import numpy as np


def sigmoid(inputs):
    pos_mask = (inputs >= 0)
    neg_mask = (inputs < 0)
    result = np.empty_like(inputs)
    result[pos_mask] = 1 / (1 + np.exp(-inputs[pos_mask]))
    result[neg_mask] = np.exp(inputs[neg_mask]) / (1 + np.exp(inputs[neg_mask]))
    return result



def ReLU(inputs: np.ndarray) -> np.ndarray:
    return np.maximum(0, inputs)


def softmax(inputs: np.ndarray) -> np.ndarray:
    exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
