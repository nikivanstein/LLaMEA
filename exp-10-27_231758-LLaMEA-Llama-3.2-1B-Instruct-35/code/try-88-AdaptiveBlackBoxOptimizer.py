import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.differential_evolution = False

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                if self.differential_evolution:
                    # Perform differential evolution
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])
                    self.func_evals -= 1
                    if self.func_evals == 0:
                        break
                else:
                    # Use local search
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])
                    self.func_evals -= 1
                    if self.func_evals == 0:
                        break

                # Refine the search space
                if self.func_evals > 0 and np.random.rand() < 0.5:
                    # Randomly shift the search space
                    idx = np.random.choice(self.dim)
                    self.func_values[idx] = func(self.func_values[idx])

# Example usage
optimizer = AdaptiveBlackBoxOptimizer(1000, 10)
def func(x):
    return np.sum(x**2)

optimizer(func)