import numpy as np
from scipy.optimize import minimize_scalar

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.explore_count = 0
        self.best_func = None
        self.best_func_val = None
        self.func_evals = []
        self.func_evals_min = []

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

    def _explore(self, func, func_evals, func_evals_min):
        if len(func_evals) > self.budget:
            func_evals = func_evals[:self.budget]
            func_evals_min = func_evals_min[:self.budget]
        else:
            func_evals = np.concatenate((func_evals, func_evals_min))
            func_evals_min = np.concatenate((func_evals, func_evals_min))
        return func_evals, func_evals_min

    def _minimize(self, func_evals):
        return minimize_scalar(func_evals, bounds=(-5.0, 5.0), method='bounded')

    def optimize(self, func):
        while self.explore_count < self.budget:
            func_evals, func_evals_min = self._explore(func, self.func_evals, self.func_evals_min)
            func_evals_min = self._minimize(func_evals_min)
            self.func_evals = np.concatenate((self.func_evals, func_evals_min))
            self.func_evals_min = np.concatenate((self.func_evals_min, func_evals_min))
            if len(self.func_evals) > self.budget:
                func_evals = self.func_evals[:self.budget]
                func_evals_min = self.func_evals_min[:self.budget]
                self.func_evals = np.concatenate((self.func_evals, func_evals_min))
                self.func_evals_min = np.concatenate((self.func_evals_min, func_evals_min))
            self.explore_count += 1
        return self.best_func

# One-line description: Hyperband is a novel metaheuristic that uses hyperband search to efficiently explore the search space of black box functions.