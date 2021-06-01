import numpy as np
from typing import Callable

class FullyConnectedLayer(object):

    '''
    # TODO:
    '''

    def __init__(self, func : Callable[[np.ndarray],np.ndarray]=None,func_prime: Callable[[np.ndarray],np.ndarray]=None) -> None:
        '''
        # TODO:
        '''
        # def sigmoid(activations : np.ndarray) -> np.ndarray:
        #     return 1.0/(1.0+np.exp(-activations))
        #
        # def sigmoid_prime(activations : np.ndarray) -> np.ndarray:
        #     return sigmoid(activations)*(1-sigmoid(activations))

        # self.neurons = size
        # self.weights = input,size
        # self.biases = size
        self.activation_function = func
        self.activation_function_prime = func_prime
        # self.activation_function = sigmoid if func is None else func
        # self.activation_function_prime = sigmoid if func_prime is None else func_prime

    # @property
    # def neurons(self) -> np.ndarray:
    #     '''
    #     # TODO:
    #     '''
    #     return self._neurons
    #
    # @neurons.setter
    # def neurons(self,value: int) -> None:
    #     '''
    #     # TODO:
    #     '''
    #     self._neurons = np.random.rand(1,value)
    #
    # @property
    # def weights(self) -> np.ndarray:
    #     '''
    #     # TODO:
    #     '''
    #     return self._weights
    #
    # @weights.setter
    # def weights(self,sizes: int) -> None:
    #     '''
    #     # TODO:
    #     '''
    #     inputs,outputs = sizes
    #     self._weights = np.random.rand(inputs,outputs)
    # @property
    # def biases(self) -> np.ndarray:
    #     '''
    #     # TODO:
    #     '''
    #     return self._biases
    #
    # @biases.setter
    # def biases(self,value: int) -> None:
    #     '''
    #     # TODO:
    #     '''
    #     self._biases = np.random.rand(value)

    @property
    def activation_function(self) -> Callable[[np.ndarray],np.ndarray]:
        '''
        # TODO:
        '''
        return self._activation_function

    @activation_function.setter
    def activation_function(self,func: Callable[[np.ndarray],np.ndarray]) -> None:
        '''
        # TODO:
        '''
        self._activation_function = func

    @property
    def activation_function_prime(self) -> Callable[[np.ndarray],np.ndarray]:
        '''
        # TODO:
        '''
        return self._activation_function_prime

    @activation_function_prime.setter
    def activation_function_prime(self,func: Callable[[np.ndarray],np.ndarray]) -> None:
        '''
        # TODO:
        '''
        self._activation_function_prime = func

    def feedforward(self, inputs, weights, biases):
        '''
        # TODO:
            Calculate : z = w*a_previous+b
            Calculate activations: a = func(z)
        '''
        # print(weights)
        # print(np.matmul(weights,inputs) + biases)
        return np.dot(weights,inputs) + biases

    def activations(self, values):
        return self.activation_function(values)

    def backpropagate(self):
        '''
        # TODO:
        '''
        print("TODO")
