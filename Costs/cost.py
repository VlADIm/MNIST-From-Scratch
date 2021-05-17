import numpy as np

# TODO: Add quadratic cost
# TODO: Add near log cost
# TODO: Document entire module

class Cost(object):
    @staticmethod
    def func():
        pass

    @staticmethod
    def delta():
        pass

class CrossEntropyCost(Cost):
    @staticmethod
    def func(a,y):
        return np.sum(np.nan_to_num(-y * np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z,a,y):
        return (a-y)
