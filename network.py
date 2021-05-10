import numpy as np
from typing import Callable

class Network:

    '''
    # TODO:
    '''

    def __init__(self,size: int, func : Callable[[np.ndarray],np.ndarray] = None,func_prime: Callable[[np.ndarray],np.ndarray] = None) -> None:
        '''
        # TODO:
        '''

        def sigmoid(activations : np.ndarray) -> np.ndarray:
            return 1.0/(1.0+np.exp(-activations))

        def sigmoid_prime(activations : np.ndarray) -> np.ndarray:
            return sigmoid(activations)*(1-sigmoid(activations))

        self.layers = size
        self.activation_function = sigmoid if func is None else func
        self.activation_function_prime = sigmoid if func_prime is None else func_prime

    @property
    def layers(self):
        '''
        # TODO:
        '''
        return self._layers

    @layers.setter
    def layers(self,value: int):
        '''
        # TODO:
        '''
        self._layers = np.random.rand(value)
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

    def feedforward(self):
        '''
        # TODO:
        '''
        print("TODO")

    def backpropagate(self):
        '''
        # TODO:
        '''
        print("TODO")
    def calculate_error():
        '''
        # TODO:
        '''
        print("TODO")
