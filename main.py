import layer
import network
import numpy as np


def main():

    test_network = network.Network()
    test_network.addLayer(layer.FullyConnectedLayer,784,100)
    test_network.addLayer(layer.FullyConnectedLayer,100,10)

    input = [np.random.randn(784,1), np.random.randn(784,1), np.random.randn(784,1)]

    # for i in input:
    #     test_network.feedforward(i)
    inputs = (np.random.randn(784,1),np.random.randn(784,1))
    outputs = (np.zeros((10,1)),np.zeros((10,1)))

    batch = zip(inputs,outputs)

    test_network.batchTraining(batch)

    print("PASSED!")
    # print(test_network.activation_function(test_network.layers))

    # inputs = np.random.rand(1,20)
    #
    # test_layer = layer.Layer(20,10)
    # print(inputs)
    # print(test_layer.feedforward(inputs))

main()
