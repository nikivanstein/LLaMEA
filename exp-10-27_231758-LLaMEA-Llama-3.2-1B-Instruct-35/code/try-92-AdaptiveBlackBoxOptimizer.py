import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Refine the search space using probabilistic search
        self.func_values = np.array([func(self.func_values)] * self.dim)
        idx = np.random.choice(self.dim, self.dim, p=self.func_values)
        self.func_values[idx] = np.random.uniform(-5.0, 5.0, self.dim)

    def refine_search(self):
        # Select the best individual using probabilistic search
        idx = np.random.choice(self.dim, self.dim, p=self.func_values)
        self.func_values[idx] = np.random.uniform(-5.0, 5.0, self.dim)