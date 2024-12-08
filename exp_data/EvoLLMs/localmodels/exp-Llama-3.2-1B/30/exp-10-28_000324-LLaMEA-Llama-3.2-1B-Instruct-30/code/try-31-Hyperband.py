import numpy as np
import random

class Hyperband:
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
        self.current_dim = 10
        self.current_budget = budget
        self.max_evals = 100

    def __call__(self, func):
        while self.explore_count < self.current_budget:
            if self.explore_count > 100:  # limit exploration to prevent infinite loop
                break
            if self.local_search:
                func_eval = func(np.random.uniform(-5.0, 5.0, self.current_dim))
            else:
                func_eval = func(np.random.uniform(-5.0, 5.0, self.current_dim))
            self.func_evals.append(func_eval)
            if self.best_func is None or np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
            self.budget_count += 1
            if self.budget_count > self.current_budget / 2:
                break
            if random.random() < self.reduction_factor:
                self.explore_count -= 1
            if random.random() < self.explore_rate:
                self.local_search = not self.local_search
            if self.explore_count >= self.max_evals:
                break
        return self.best_func

# One-line description: Hyperband uses a combination of hyperband search and local search to efficiently explore the search space of black box functions.
# Code: 