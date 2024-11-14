import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))  # Initialize population using Latin Hypercube Sampling

    def __call__(self, func):
        # Implementation of Differential Evolution algorithm here to optimize the black box function
        pass