import numpy as np

class AdaptiveProbabilisticMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pa = 0.25

    def __call__(self, func):
        for _ in range(self.budget):
            self.pa = np.clip(self.pa * 1.05, 0, 1)
            # Use the probability self.pa for accepting new solutions
            # Optimization logic here
        return best_solution