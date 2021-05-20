import numpy as np
from typing import Callable
import layer
import Costs.cost as cost
import random

class Network:

    '''
    # TODO:
    '''

    def __init__(self, func : Callable[[np.ndarray],np.ndarray] = None,func_prime: Callable[[np.ndarray],np.ndarray] = None) -> None:
        '''
        # TODO:
        '''

        def sigmoid(activations : np.ndarray) -> np.ndarray:
            return 1.0/(1.0+np.exp(-activations))

        def sigmoid_prime(activations : np.ndarray) -> np.ndarray:
            return sigmoid(activations)*(1-sigmoid(activations))

        self.layer_weights = []
        self.layer_biases = []
        self.layer_objects = []
        self.activation_function = sigmoid if func is None else func
        self.activation_function_prime = sigmoid if func_prime is None else func_prime

    @property
    def layer_weights(self) -> list:
        '''
        # TODO:
        '''
        return self._layer_weights

    @layer_weights.setter
    def layer_weights(self,value: list) -> None:
        '''
        # TODO:
        '''
        self._layer_weights = value

    @property
    def layer_biases(self) -> list:
        '''
        # TODO:
        '''
        return self._layer_biases

    @layer_biases.setter
    def layer_biases(self,value: list) -> None:
        '''
        # TODO:
        '''
        self._layer_biases = value

    @property
    def layer_activations(self) -> list:
        '''
        # TODO:
        '''
        return self._layer_activations

    @layer_activations.setter
    def layer_activations(self,value: list) -> None:
        '''
        # TODO:
        '''
        self._layer_activations = value
    @property
    def layer_objects(self) -> list:
        '''
        # TODO:
        '''
        return self._layer_objects

    @layer_objects.setter
    def layer_objects(self,value: list) -> None:
        '''
        # TODO:
        '''
        self._layer_objects = value

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

    def addLayer(self, layer_object, input: int, output: int) -> None:
        '''
        # TODO:
        '''
        self.layer_objects.append(layer_object(self.activation_function,self.activation_function_prime))
        self.layer_weights.append(np.random.randn(output,input)/np.sqrt(input))
        self.layer_biases.append(np.random.randn(output,1))
        print("TODO: check that new layers match neighboring layers")

    def validateLayers(self):
        '''
        # TODO:
        '''

        print("UNFINISHED METHOD: Network.validateLayers()")

    def feedforward(self, input: np.ndarray) -> np.ndarray:
        '''
        # TODO:
        '''
        for layer_object,layer_weight,layer_biase in zip(self.layer_objects,self.layer_weights, self.layer_biases):
            input = layer_object.feedforward(input, layer_weight, layer_biase)
        print("UNFINISHED METHOD: Network.feedforward()")

    def calculateError(self):
        '''
        # TODO:
        '''
        print("UNFINISHED METHOD: Network.calculateError()")

    def run(self, input):
        '''
        # TODO:
        '''
        print("UNFINISHED METHOD: Network.run()")

    def stochasticGradientDescent(self, training_dataset, batch_size = 1, epoch= 10, learning_rate = 0.05, regularization_rate = 0.0):
        '''
        # TODO:
        '''
        training = list(training_dataset)
        training_set_size = len(training)

        for i in range(epoch):
            print("Training epoch %s of %s" %(i+1, epoch))
            random.shuffle(training)
            batches = [training[j:j+batch_size] for j in range(0,training_set_size,batch_size)]

            for batch in batches:
                self.batchTraining(batch,learning_rate,regularization_rate,training_set_size)

        # print("UNFINISHED METHOD: Network.stochasticGradientDescent()")


    def batchTraining(self, batch, learning_rate, regularization_rate, training_set_size):
        '''
        # TODO:
        batch is (inputs, outputs)
        '''

        partial_of_weights = [np.zeros(weights.shape) for weights in self.layer_weights]
        partial_of_biases = [np.zeros(biases.shape) for biases in self.layer_biases]

        for input, output in batch:
            delta_partial_of_weights, delta_partial_of_biases = self.backpropagate(input,output)
            partial_of_weights = [weight+delta_weight for weight, delta_weight in zip(partial_of_weights, delta_partial_of_weights)]
            partial_of_biases = [biase+delta_biase for biase, delta_biase in zip(partial_of_biases, delta_partial_of_biases)]

        self.layer_weights = [ (1 - learning_rate*regularization_rate/training_set_size) * weight - learning_rate/len(batch) * delta_weight for weight, delta_weight in zip(self.layer_weights, partial_of_weights)]
        self.layer_biases = [biase - (learning_rate/len(batch)) * delta_biase for biase, delta_biase in zip(self.layer_biases, partial_of_biases)]
        # print("UNFINISHED METHOD: Network.batchTraining()")


    def backpropagate(self, input, output):
        '''
        # TODO:
        '''
        partial_of_weights = [np.zeros(weights.shape) for weights in self.layer_weights]
        partial_of_biases = [np.zeros(biases.shape) for biases in self.layer_biases]
        layer_activations = [input]
        layer_values = []

        for layer, weight, biase in zip(self.layer_objects, self.layer_weights, self.layer_biases):
            layer_values.append(layer.feedforward(input, weight, biase))
            layer_activations.append(layer.activations(layer_values[-1]))
            input = layer_activations[-1]

        delta = cost.CrossEntropyCost.delta(layer_values[-1],layer_activations[-1],output)

        partial_of_weights[-1] = np.dot(delta, layer_activations[-2].transpose())
        partial_of_biases[-1] = delta

        for layer in range(2,len(self.layer_objects)+1):
            delta = np.dot(self.layer_weights[-layer+1].transpose(),delta) * (self.layer_objects[-layer].activation_function(layer_values[-layer]))
            partial_of_biases[-layer] = delta
            partial_of_weights[-layer] = np.dot(delta, layer_activations[-layer-1].transpose())


        return (partial_of_weights, partial_of_biases)
        # print("UNFINISHED METHOD: Network.backpropagate()")

        # cost = 1/n
        #
        #  = 1 / n * (sum(activations[-1] * (activations-)))

        # calculate cost
        # print(type(cost.CrossEntropyCost.func(a,y))


        # for input, output in batch:



        # NEED Z VALUE FOR ALL LAYERS
        # NEED ACTIVATIONS FOR ALL LAYERS

        # NEED TO CALCULATE DELTA

        # NEED TO CALCULATE PARTIAL OF b
        # NEED TO CALCULATE PARTIAL OF w
