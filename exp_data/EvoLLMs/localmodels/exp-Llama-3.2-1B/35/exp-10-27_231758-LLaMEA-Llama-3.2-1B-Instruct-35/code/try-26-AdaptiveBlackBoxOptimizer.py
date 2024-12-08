import numpy as np

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

        # Refine the strategy by changing the probability of the current individual
        # This is based on the idea of "crossover" and "mutation" in genetic algorithms
        # We will use a simple version of this strategy, where the probability is 0.5
        if np.random.rand() < 0.35:
            idx = np.random.choice(self.dim, p=[0.6, 0.4])
            self.func_values[idx] = func(self.func_values[idx])

# One-line description: AdaptiveBlackBoxOptimizer refines its strategy using a simple crossover and mutation approach
# Code: 