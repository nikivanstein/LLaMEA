import numpy as np
import random
from scipy.optimize import minimize

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.fidelity_levels = [1, 10, 100]
        self.fidelity_map = {1: self.eval_fidelity, 10: self.eval_fidelity_10, 100: self.eval_fidelity_100}

    def __call__(self, func):
        self.fidelity_map[random.choice(self.fidelity_levels)](func)
        self.budget -= 1
        if self.budget == 0:
            return self.get_best_solution(func)

    def eval_fidelity(self, func):
        x = np.random.uniform(-5.0, 5.0, size=self.dim)
        y = func(x)
        return y

    def eval_fidelity_10(self, func):
        x = np.random.uniform(-5.0, 5.0, size=self.dim)
        y = func(x)
        return y * 10

    def eval_fidelity_100(self, func):
        x = np.random.uniform(-5.0, 5.0, size=self.dim)
        y = func(x)
        return y * 100

    def get_best_solution(self, func):
        def neg_func(x):
            return -func(x)

        res = minimize(neg_func, np.random.uniform(-5.0, 5.0, size=self.dim), method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim)
        return res.x

# Example usage:
def example_func(x):
    return np.sum(x**2)

metaheuristic = Metaheuristic(budget=10, dim=2)
metaheuristic(example_func)