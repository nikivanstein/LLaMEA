import numpy as np
import random

class NonLocalDABU:
    def __init__(self, budget, dim, alpha=0.5, beta=0.8, min_alpha=0.01, max_alpha=0.9, min_beta=0.01, max_beta=0.1):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.current_alpha = alpha
        self.current_beta = beta
        self.convergence_rate = 0.8
        self.convergence_threshold = 0.9
        self.search_space_size = 100
        self.convergence_history = []

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            if self.func_evaluations / self.budget > self.convergence_rate:
                self.alpha *= self.beta
                if self.alpha < self.min_alpha:
                    self.alpha = self.min_alpha
                if self.alpha > self.max_alpha:
                    self.alpha = self.max_alpha
            if self.func_evaluations / self.budget > self.convergence_threshold:
                self.beta *= self.alpha
                if self.beta < self.min_beta:
                    self.beta = self.min_beta
                if self.beta > self.max_beta:
                    self.beta = self.max_beta
            # Non-Local Search
            for i in range(self.dim):
                for j in range(self.dim):
                    if random.random() < self.alpha:
                        self.search_space[i] = np.random.uniform(-5.0, 5.0)
                        self.search_space[j] = np.random.uniform(-5.0, 5.0)
            # Evaluate the function with the new search space
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            # Store the convergence history
            self.convergence_history.append(np.abs(func_value))
            if len(self.convergence_history) > 10:
                self.convergence_history.pop(0)
            # Non-Local Search
            for k in range(self.dim):
                for l in range(self.dim):
                    if random.random() < self.alpha:
                        self.search_space[k] = np.random.uniform(-5.0, 5.0)
                        self.search_space[l] = np.random.uniform(-5.0, 5.0)
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

# Description: Novel metaheuristic algorithm using Non-Local Search with Adaptation to Convergence Rate
# Code: 
# ```python
# ```python
# ```python
# # Refine the strategy by changing the individual lines
def refine_strategy(nonlocal_dabu):
    # Increase the budget to 5000
    nonlocal_dabu.budget = 5000
    # Increase the number of dimensions to 3
    nonlocal_dabu.dim = 3
    # Increase the alpha value to 0.7
    nonlocal_dabu.alpha = 0.7
    # Decrease the convergence threshold to 0.8
    nonlocal_dabu.convergence_threshold = 0.8
    # Increase the number of convergence history entries to 20
    nonlocal_dabu.convergence_history = []
    return nonlocal_dabu

nonlocal_dabu = refine_strategy(nonlocal_dabu)
print(nonlocal_dabu(test_function))  # prints a random value between -10 and 10