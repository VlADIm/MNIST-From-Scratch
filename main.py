import layer
import network
import numpy as np


def main():
    test_network = network.Network(20)
    print(test_network.activation_function(test_network.layers))

    test_layer = layer.Layer(20)
    print(test_layer.inputs.shape)

main()
