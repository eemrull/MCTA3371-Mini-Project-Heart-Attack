import numpy as np


class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation: callable) -> None:
        self.weights = np.random.uniform(low=-1,
                                         high=1,
                                         size=(n_inputs, n_neurons))
        # self.biases = np.zeros((1, n_neurons))
        self.biases = np.random.uniform(low=0,
                                         high=1,
                                         size=(1, n_neurons))
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(inputs, self.weights) + self.biases)


class NN:
    def __init__(self) -> None:
        self.layers: list[Layer] = []

    def add(self, layer: Layer):
        self.layers.append(layer)
        return self

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs
