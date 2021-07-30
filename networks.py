import json
import numpy as np
from activations import SigmoidActivation
from costs import CrossEntropyCost


class Network(object):
    '''
    # TODO:
    '''

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.activation = SigmoidActivation
        self.cost = CrossEntropyCost
        self.initializeValues()

    def initializeValues(self):
        # Set all layer biases excluding
        '''
        Initialization function used to initialize all network biases and weights
        '''
        self.layer_biases = [np.random.randn(layer,1) for layer in self.layers[1:]]
        self.layer_weights = [np.random.randn(layer_prime,layer)/np.sqrt(layer) for layer,layer_prime in zip(self.layers[:-1],self.layers[1:])]

    ############################################################################
    ############################################################################
    ############################################################################

    def feedforward(self,input):
        '''
        Feed forward input through entire network and recieve output
        '''
        for weight, biase in zip(self.layer_weights,self.layer_biases):
            input = self.activation.func(np.dot(weight,input)+biase)
        return input

    ############################################################################
    ############################################################################
    ############################################################################

    def runStatistics(self, data, regularization_rate):
        training_cost = self.calculateTrainingCost(data,regularization_rate)
        training_accuracy = self.calculateTrainingAccuracy(data,regularization_rate)
        print("Training Cost: %f" %(training_cost))
        print("Training Accuracy: %f" %(training_accuracy))

    def calculateTrainingAccuracy(self, data, regularization_rate):
        results = [(np.argmax(self.feedforward(input)),np.argmax(output)) for input,output in data]
        result_accuracy = sum(int(result == output) for result, output in results)
        return result_accuracy

    def calculateError(self, data):
        raise Exception("METHOD INCOMPLETE")

    def calculateTrainingCost(self, data, regularization_rate):
        cost = 0.0
        for input, output in data:
            result = self.feedforward(input)
            cost += self.cost.func(result,output)/len(data)
            cost += 0.5*(regularization_rate/len(data))*sum(np.linalg.norm(weight)**2 for weight in self.layer_weights)
        return cost

    def showStatistics(self):
        raise Exception("METHOD INCOMPLETE")

    ############################################################################
    ############################################################################
    ############################################################################

    def stochasticGradientDescent(self, training_data, epochs, batch_size, learning_rate, regularization_rate):
        training_data = list(training_data)

        for i in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[j:j+batch_size] for j in range(0,len(training_data),batch_size)]
            for batch in mini_batches:
                self.miniBatchTraining(batch, learning_rate, regularization_rate, len(training_data))


            self.runStatistics(training_data,regularization_rate)
        # TODO: ADD HEURISTICS TO stochasticGradientDescent METHOD TO QUANTIFY RESULTS
        # raise Exception("METHOD INCOMPLETE")

    def miniBatchTraining(self, batch, learning_rate, regularization_rate, data_length):
        '''
        # TODO:
        '''
        partial_of_weights = [np.zeros(weight.shape) for weight in self.layer_weights]
        partial_of_biases = [np.zeros(biase.shape) for biase in self.layer_biases]

        for input, output in batch:
            change_of_weights, change_of_biases = self.backpropagate(input,output)
            partial_of_weights = [weight+change for weight, change in zip(partial_of_weights, change_of_weights)]
            partial_of_biases = [biase+change for biase, change in zip(partial_of_biases, change_of_biases)]

        self.layer_weights = [(1-learning_rate*(regularization_rate/data_length))*weight-(learning_rate/len(batch))*partial for weight, partial in zip(self.layer_weights,partial_of_weights)]
        self.layer_biases = [biase - (learning_rate*(regularization_rate/data_length))*partial for biase, partial in zip(self.layer_biases,partial_of_biases)]

    def backpropagate(self,input,output):
        '''
        # TODO:
        '''
        # Initialize local variables
        partial_of_weights = [np.zeros(weight.shape) for weight in self.layer_weights]
        partial_of_biases = [np.zeros(biase.shape) for biase in self.layer_biases]
        vectors = []
        activations = []

        # Input
        active = input
        activations.append(active)

        # Start feedforward with input given and record all vectors and activations in order
        for weight, biase in zip(self.layer_weights,self.layer_biases):
            result = np.dot(weight,active)+biase
            active = self.activation.func(result)
            vectors.append(result)
            activations.append(active)

        # Start backward Pass
        change = (self.cost).delta(vectors[-1],activations[-1],output)
        partial_of_weights[-1] = np.dot(change,activations[-2].transpose())
        partial_of_biases[-1] = change

        for i in range(2,self.num_layers):
            change = np.dot(self.layer_weights[-i+1].transpose(),change)*(self.activation).prime(vectors[-i])
            partial_of_weights[-i] = np.dot(change, activations[-i-1].transpose())
            partial_of_biases[-i] = change
        return (partial_of_weights,partial_of_biases)

    ############################################################################
    ############################################################################
    ############################################################################

    def load(self):
        raise Exception("METHOD INCOMPLETE")

    def save(self, filename):
        data = {"weights": [weights.tolist() for weights in self.layer_weights],
                "biases": [biases.tolist() for biases in self.layer_biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        # raise Exception("METHOD INCOMPLETE")


    ############################################################################
    ############################################################################
    ############################################################################
