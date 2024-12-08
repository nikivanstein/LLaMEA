import numpy as np

class DynamicAdaptiveEDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = np.random.uniform(0.2, 0.4)
        self.step_size = 0.01

    def __call__(self, func):
        for _ in range(self.budget):
            self.mutation_rate = self.mutation_rate * np.exp(-self.step_size * _)
            # Update parameter values based on dynamic self-adaptive strategy
            # Perform evolution and optimization steps
        return self.get_global_best()