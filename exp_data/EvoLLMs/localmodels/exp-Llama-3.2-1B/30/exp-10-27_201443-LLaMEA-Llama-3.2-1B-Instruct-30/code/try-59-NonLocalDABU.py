import numpy as np
import random

class NonLocalDABU:
    def __init__(self, budget, dim, alpha=0.5, beta=0.8):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta
        self.best_func_value = None
        self.best_func_evaluations = 0
        self.current_func_value = None
        self.current_func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            self.func_evaluations += 1
            if self.current_func_value is not None and np.abs(self.current_func_value) < 1e-6:
                break
            if self.func_evaluations / self.budget > 0.8:
                self.alpha *= self.beta
            if self.func_evaluations / self.budget > 0.9:
                self.beta *= self.alpha
            if self.func_evaluations / self.budget > 0.95:
                # Refine the strategy
                self.alpha *= 0.7
                self.beta *= 0.3
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
            self.current_func_value = func(self.search_space)
            self.current_func_evaluations += 1
            # Update the best function value and evaluations
            if self.current_func_value < self.best_func_value or self.best_func_evaluations == 0:
                self.best_func_value = self.current_func_value
                self.best_func_evaluations = self.current_func_evaluations
        return self.current_func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

nonlocal_dabu = NonLocalDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(nonlocal_dabu(test_function))  # prints a random value between -10 and 10