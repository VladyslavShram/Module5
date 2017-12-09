import numpy as np
import math


class Kernel(object):
    """Check kernels here https://en.wikipedia.org/wiki/Support_vector_machine"""
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma=1):
    	return lambda x, y: math.exp(-np.linalg.norm(x - y)**2 / sigma**2)
        