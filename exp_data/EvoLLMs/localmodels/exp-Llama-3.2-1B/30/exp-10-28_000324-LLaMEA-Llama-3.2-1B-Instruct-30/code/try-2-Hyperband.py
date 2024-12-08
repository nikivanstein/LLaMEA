import numpy as np
import random
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
        self.func_evaluations = []

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

    def update(self, func):
        while self.explore_count < self.budget:
            if self.explore_count > 100:  # limit exploration to prevent infinite loop
                break
            func_eval = func(np.random.uniform(-5.0, 5.0, self.dim))
            self.func_evals.append(func_eval)
            if self.best_func is None or np.abs(func_eval - self.best_func_val) > np.abs(func_eval - self.best_func):
                self.best_func = func_eval
                self.best_func_val = func_eval
            self.explore_count += 1
        self.func_evaluations.append(func)

    def __str__(self):
        return f"Hyperband: {self.__class__.__name__} (Score: {self.score:.4f})"

    def score(self):
        return self.best_func_val

# One-line description: Hyperband is a novel metaheuristic that uses hyperband search to efficiently explore the search space of black box functions.

# Hyperband Algorithm with Refinement Strategy
class HyperbandRefined(Hyperband):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.refine_rate = 0.05

    def refine(self):
        # Evaluate the best function up to the refine_rate
        self.update(self.best_func)
        # Refine the search space by evaluating the best function at the refine_rate intervals
        intervals = np.linspace(-5.0, 5.0, 10)
        for i in range(10):
            self.update(self.best_func)
            # Evaluate the best function at the refine_rate intervals
            intervals[i] = self.best_func_val + i * (5.0 - (-5.0)) / 10.0
        # Refine the search space
        self.explore_rate = self.refine_rate
        self.best_func = None
        self.best_func_val = None
        self.func_evals = []
        self.func_evaluations = []

    def __call__(self, func):
        refined_func = super().__call__(func)
        refined_func_evals = []
        for i in range(10):
            refined_func_evals.append(redefined_func(np.random.uniform(-5.0, 5.0, self.dim)))
        refined_func.update(redefined_func_evals)
        return refined_func

# Example usage:
func = lambda x: x**2
hyperband = Hyperband(100, 2)
print(hyperband.score())

hyperband_refined = HyperbandRefined(100, 2)
print(hyperband_refined.score())

hyperband_refined_refined = HyperbandRefined(100, 2)
print(hyperband_refined_refined.score())
