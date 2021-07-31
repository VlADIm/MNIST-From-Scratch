import numpy as np

class Cost(object):
    '''
    # TODO:
    '''
    @staticmethod
    def func():
        pass

    @staticmethod
    def delta():
        pass


class CrossEntropyCost(Cost):
    '''
    # TODO:
    '''
    @staticmethod
    def func(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z,a,y):
        return (a-y)
