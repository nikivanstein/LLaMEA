import numpy as np
from scipy.optimize import minimize
import random

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.adaptive_search = False
        self.adaptive_strategy = None

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def update_adaptive_search(self):
        if random.random() < 0.2:
            if self.adaptive_search:
                self.adaptive_strategy = random.choice([0, 1, 2])
            else:
                self.adaptive_strategy = 0

    def __call__(self, func):
        self.update_adaptive_search()
        if self.adaptive_strategy == 0:
            return wrapper(self.search_space[0])
        elif self.adaptive_strategy == 1:
            return wrapper(self.search_space[1])
        else:
            return wrapper(self.search_space[2])