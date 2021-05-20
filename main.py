import layer
import network
import numpy as np
import mnist

def loadMNISTDataset():

    def vectorizeLabel(label: int):
        return_val = np.zeros((10,1))
        return_val[label] = 1
        return return_val

    data = mnist.MNIST(".")
    training_images,training_labels = data.load_training()
    testing_images, testing_labels = data.load_testing()

    labels = np.vectorize(vectorizeLabel)

    training_images = [np.asarray(images).reshape(784,1) for images in training_images]
    training_labels = [labels(number) for number in training_labels]

    testing_images = [np.asarray(images) for images in testing_images]
    testing_labels = [labels(number) for number in testing_labels]

    validation_images = training_images[-10000:]
    validation_labes = training_labels[-10000:]

    training_images = training_images[:-10000]
    training_labels = training_labels[:-10000]

    return (zip(training_images,training_labels), zip(validation_images, validation_labes), zip(testing_images, testing_labels))

def main():

    train,validate,test = loadMNISTDataset()

    test_network = network.Network()
    test_network.addLayer(layer.FullyConnectedLayer,784,100)
    test_network.addLayer(layer.FullyConnectedLayer,100,10)

    test_network.stochasticGradientDescent(train, batch_size=30, epoch=30);


    # input = [np.random.randn(784,1), np.random.randn(784,1), np.random.randn(784,1)]


    # dataset = loadMNISTDataset()

    # for i in input:
    #     test_network.feedforward(i)
    # inputs = (np.random.randn(784,1),np.random.randn(784,1))
    # outputs = (np.zeros((10,1)),np.zeros((10,1)))
    #
    # batch = zip(inputs,outputs)




    print("PASSED!")
    # print(test_network.activation_function(test_network.layers))

    # inputs = np.random.rand(1,20)
    #
    # test_layer = layer.Layer(20,10)
    # print(inputs)
    # print(test_layer.feedforward(inputs))

#
# test = np.zeros(10)
#
# test[4] = 1
#
# print(test)

# i,j,k = loadMNISTDataset()
# print(len(list(i)))



main()
