import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(10)]

    def __call__(self, func):
        # Implementation of Differential Evolution algorithm here to optimize the black box function
        pass