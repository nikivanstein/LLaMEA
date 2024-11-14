import numpy as np

class NovelDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.strategy = np.random.uniform(0.5, 1.0, (budget, dim))  # Initialize mutation strategy
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.dim):
                # Update strategy based on the population
                self.strategy[:, i] = np.maximum(0.1, self.strategy[:, i] - 0.01) if np.mean(self.population[:, i]) < 0 else np.minimum(2.0, self.strategy[:, i] + 0.01)
                # Your optimization logic here
            pass