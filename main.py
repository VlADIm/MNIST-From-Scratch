import networks
import mnist_loader


def main():

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    temp = networks.Network([784,100,10])
    temp.stochasticGradientDescent(training_data,30,20,0.5,0.1)
    temp.save('network.json')

main()
