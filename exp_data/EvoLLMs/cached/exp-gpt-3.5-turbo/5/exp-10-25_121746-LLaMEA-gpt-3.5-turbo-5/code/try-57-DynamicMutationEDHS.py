import numpy as np

class DynamicMutationEDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = np.random.uniform(0.2, 0.4)

    def __call__(self, func):
        for _ in range(self.budget):
            self.mutation_rate = self.mutation_rate * np.exp(-0.01 * _)
            # Perform EDHS optimization steps
            # Update population based on mutation rate
        return global_best_solution