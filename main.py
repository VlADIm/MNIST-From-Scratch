import networks
import mnist_loader


def main():

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    temp = networks.Network([784,100,10])
    # training_data, epochs, batch_size, learning_rate, regularization_rate
    temp.stochasticGradientDescent(training_data,30,10,0.1,5.0)
    temp.save('new_network.json')

main()
