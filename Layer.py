import numpy as np

class layer:

    neurons = np.array([])

    def __init__(self, size: int):
        neurons = np.ones(size)
