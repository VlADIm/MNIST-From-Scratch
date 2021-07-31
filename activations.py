import numpy as np

class Activation(object):
    @staticmethod
    def func(input):
        pass
    @staticmethod
    def prime(input):
        pass


class SigmoidActivation(Activation):
    @staticmethod
    def func(input):
        return 1.0/(1.0+np.exp(-input))
    @staticmethod
    def prime(input):
        return SigmoidActivation.func(input)*(1-SigmoidActivation.func(input))
