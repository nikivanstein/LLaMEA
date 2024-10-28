import numpy as np
import random

class AdaptiveHyperbandAOACS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.explore_count = 0
        self.best_func = None
        self.best_func_val = None
        self.func_evals = []
        self.budget_count = 0
        self.reduction_factor = 0.3
        self.local_search = True
        self.deterministic = False
        self.explore_strategy = 'uniform'
        self.local_search_strategy = 'random'

    def __call__(self, func):
        while self.explore_count < self.budget:
            if self.explore_strategy == 'uniform':
                func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
            elif self.explore_strategy == 'adaptive':
                if self.explore_count / self.budget < 0.3:
                    func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
                else:
                    func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
            else:
                raise ValueError("Invalid exploration strategy. Choose from 'uniform' or 'adaptive'.")

            self.func_evals.append(func_eval)
            if self.best_func is None or np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
            self.budget_count += 1
            if self.budget_count > self.budget / 2:
                break
            if random.random() < self.reduction_factor:
                self.explore_count -= 1
            if random.random() < self.explore_rate:
                self.local_search = not self.local_search
        return self.best_func

# One-line description: Adaptive Hyperband Algorithm with Adaptive Exploration Strategy
# Code: 
# ```python
# Adaptive Hyperband Algorithm with Adaptive Exploration Strategy
# ```
# ```python
def adaptive_hyperbandAOACS(budget, dim):
    adapt_hyperband = AdaptiveHyperbandAOACS(budget, dim)
    return adapt_hyperband(adaptive_hyperbandAOACS)

# Test the function
budget = 100
dim = 5
best_func, best_func_val = adaptive_hyperbandAOACS(budget, dim)
print(f"Best function: {best_func}, Best function value: {best_func_val}")