import numpy as np
from scipy.optimize import differential_evolution

class NonLocalDABU:
    def __init__(self, budget, dim, alpha=0.5, beta=0.8):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            # Adapt the strategy to convergence rate
            if self.func_evaluations / self.budget > 0.8:
                self.alpha *= self.beta
            if self.func_evaluations / self.budget > 0.9:
                self.beta *= self.alpha
            # Non-Local Search
            for i in range(self.dim):
                for j in range(self.dim):
                    if np.random.rand() < self.alpha:
                        self.search_space[i] = np.random.uniform(-5.0, 5.0)
                        self.search_space[j] = np.random.uniform(-5.0, 5.0)
            # Evaluate the function with the new search space
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

nonlocal_dabu = NonLocalDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(nonlocal_dabu(test_function))  # prints a random value between -10 and 10