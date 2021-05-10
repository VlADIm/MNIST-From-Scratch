import numpy as np
from typing import Callable

class Layer:

    '''
    # TODO:
    '''

    def __init__(self,size: int, func : Callable[[np.ndarray],np.ndarray]=None,func_prime: Callable[[np.ndarray],np.ndarray]=None) -> None:
        '''
        # TODO:
        '''
        self.inputs = size
        self.weights = size
        self.biases = size
        self.activation_function = func
        self.activation_function_prime = func_prime

    @property
    def inputs(self) -> np.ndarray:
        '''
        # TODO:
        '''
        return self._inputs

    @inputs.setter
    def inputs(self,value: int) -> None:
        '''
        # TODO:
        '''
        self._inputs = np.random.rand(1,value)

    @property
    def weights(self) -> np.ndarray:
        '''
        # TODO:
        '''
        return self._weights

    @weights.setter
    def weights(self,value: int) -> None:
        '''
        # TODO:
        '''
        self._weights = np.random.rand(1,value)
    @property
    def biases(self) -> np.ndarray:
        '''
        # TODO:
        '''
        return self._biases

    @biases.setter
    def biases(self,value: int) -> None:
        '''
        # TODO:
        '''
        self._biases = np.random.rand(value)

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
            Calculate : z = w*a_previous+b
            Calculate activations: a = func(z)
        '''

        np.matmul(inputs,weights)

        print("TODO")

    def backpropagate(self):
        '''
        # TODO:
        '''
        print("TODO")
