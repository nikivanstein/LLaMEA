import numpy as np
import random

class AdaptiveHyperband:
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
        self.reduction_factor_count = 0
        self.local_search_count = 0
        self.score = 0.0

    def __call__(self, func):
        while self.explore_count < self.budget:
            if self.explore_count > 100:  # limit exploration to prevent infinite loop
                break
            if self.local_search:
                func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
            else:
                func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
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
            self.reduction_factor_count += 1
            if self.reduction_factor_count > 10:
                self.reduction_factor -= 0.1
                self.reduction_factor_count = 0
            if random.random() < self.explore_rate:
                self.local_search_count += 1
                if self.local_search_count > 10:
                    self.local_search = not self.local_search
                    self.local_search_count = 0
        return self.best_func

# One-line description: Adaptive Hyperband uses adaptive reduction factor and local search to efficiently explore the search space of black box functions.