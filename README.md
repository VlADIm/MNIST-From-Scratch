# Design Document
##### MNIST Prototype
###### Author: Vladimir Malcevic
---
__Project Goal:__ Write a MNIST classification neural network without using libraries such as PyTorch or TensorFlow to learn machine learning algorithms and processes.

__Further Details:__
- Project will use following online resources to research and implement algorithms
    - [Neural Networks and Deep Learning Online Textbook](http://neuralnetworksanddeeplearning.com/)
- Specific Goals
    - Achieve over 90% accuracy on the test data set
    - Utilize CUDA architecture to accelerate learning

__Code Outline__:
- __mnist_loader.py__ : _Python file that loads and parses the MNIST dataset into training, validating, and testing datasets_
- __network.py__ : _Python file that contains the network class that handles training, classifying, saving, and loading neural networks_
- __layer.py__ : _Python file that contains the layer classes that handle specific layer behaviors_
- __costs.py__ : _Python file that contains all cost functions used by neural networks_
- __activations.py__ : _Python file that contains all activation functions used by neural networks_
- __main.py__ : _Python file that contains the implementation of the network class to train, run, and output the results_
