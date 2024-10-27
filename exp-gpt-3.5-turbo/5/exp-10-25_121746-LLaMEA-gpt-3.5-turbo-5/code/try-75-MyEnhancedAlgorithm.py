import numpy as np

class MyEnhancedAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Initialize mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            if np.random.rand() < 0.05:  # Adjust mutation rate with a probability
                self.mutation_rate = np.random.uniform(0.1, 0.5)
            # Implement your own optimization strategy here using the mutation rate
        return global_best_solution