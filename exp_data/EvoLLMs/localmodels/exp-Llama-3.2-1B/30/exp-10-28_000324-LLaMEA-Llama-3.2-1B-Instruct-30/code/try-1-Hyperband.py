import numpy as np
import random
import math

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.explore_count = 0
        self.best_func = None
        self.best_func_val = None
        self.func_evals = []

    def __call__(self, func):
        while self.explore_count < self.budget:
            if self.explore_count > 100:  # limit exploration to prevent infinite loop
                break
            func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
            self.func_evals.append(func_eval)
            if self.best_func is None or np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
        return self.best_func

    def __str__(self):
        return f"Hyperband: (Score: {self.score:.6f}, Explored: {self.explored:.2f})"

    def update_explore_rate(self, new_rate):
        if new_rate > self.explore_rate:
            self.explore_rate = new_rate

class HyperbandHyperband(Hyperband):
    def __init__(self, budget, dim, alpha, beta):
        super().__init__(budget, dim)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        while self.explore_count < self.budget:
            if self.explore_count > 100:  # limit exploration to prevent infinite loop
                break
            func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
            self.func_evals.append(func_eval)
            if self.best_func is None or np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
        # Refine the strategy using the new alpha and beta values
        self.update_explore_rate(self.alpha * self.beta)

# One-line description: HyperbandHyperband is a novel metaheuristic that combines hyperband search with adaptive exploration rates to efficiently explore the search space of black box functions.