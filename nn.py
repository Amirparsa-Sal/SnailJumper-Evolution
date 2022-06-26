import numpy as np
from abc import ABC, abstractmethod
from typing import List

class ActicationFunction(ABC):

    @abstractmethod
    def activation(self, X):
        pass

class Sigmoid(ActicationFunction):

    def activation(self, X):
        return 1 / (1 + np.exp(-X))

class Tanh(ActicationFunction):

    def activation(self, X):
        return np.tanh(X)

class ReLU(ActicationFunction):
    
    def activation(self, X):
        return np.maximum(0, X)
        
class LeakyRelu(ActicationFunction):

    def activation(self, X):
        return np.maximum(0.01 * X, X)

class ActivationFunctionFactory:

    activations = {
        "sigmoid": Sigmoid(),
        "tanh": Tanh(),
        "relu": ReLU(),
        "leaky_relu": LeakyRelu()
    }

    @classmethod
    def get_activation_function(self, name: str):
        if name in self.activations:
            return self.activations[name]
        raise ValueError("Unknown activation function: " + name)

class NeuralNetwork:

    def __init__(self, layer_sizes: List[int], hidden_activation: str = 'sigmoid', last_layer_activation: str = 'sigmoid'):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        :param activation_function: Activation function of the network.
        """
        self.hidden_activation = ActivationFunctionFactory.get_activation_function(hidden_activation)
        self.last_layer_activation = ActivationFunctionFactory.get_activation_function(last_layer_activation)
        self.layer_sizes = layer_sizes
        self.weights = []
        for i in range (len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]))
        self.biases = []
        for i in range (len(layer_sizes) - 1):
            self.biases.append(np.zeros((layer_sizes[i + 1], 1)))
        self.a = []
        for i in range (len(layer_sizes)):
            self.a.append(np.zeros((layer_sizes[i], 1)))


    def activation(self, X):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param X: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        return self.hidden_activation.activation(X)

    def forward(self, X):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        self.a[0] = X
        for i in range (len(self.a) - 2):
            self.a[i + 1] = self.activation(np.dot(self.weights[i], self.a[i]) + self.biases[i])
        self.a[-1] = self.last_layer_activation.activation(np.dot(self.weights[-1], self.a[-2]) + self.biases[-1])
        return self.a[-1]