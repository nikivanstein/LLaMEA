import numpy as np
from scipy.stats import logistic

class ImprovedDynamicInertiaPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize particles with chaotic initialization
        def chaotic_map(x, a=3.999):
            return logistic.cdf(a * x * (1 - x))
        
        population = np.array([chaotic_map(np.random.rand(self.dim)) * 10 - 5 for _ in range(self.dim)])

        # Rest of the implementation remains unchanged from the original PSO algorithm
        pass