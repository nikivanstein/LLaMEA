import numpy as np
import random
import copy
from scipy.optimize import minimize

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class Hyperband:
    def __init__(self, budget, dim, num_searches):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.num_searches = num_searches

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            new_individual = copy.deepcopy(self.search_space)
            if random.random() < 0.5:
                new_individual = random.uniform(self.search_space)
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class EvolutionaryHyperband(HEBBO):
    def __init__(self, budget, dim, num_searches, p1, p2):
        super().__init__(budget, dim)
        self.p1 = p1
        self.p2 = p2
        self.num_searches = num_searches

    def __call__(self, func):
        new_individual = copy.deepcopy(self.search_space)
        if random.random() < self.p1:
            new_individual = random.uniform(self.search_space)
        elif random.random() < self.p2:
            new_individual = random.uniform(self.search_space)
        else:
            new_individual = copy.deepcopy(self.search_space)
            if random.random() < 0.5:
                new_individual = random.uniform(self.search_space)
            else:
                new_individual = np.linspace(-5.0, 5.0, self.dim)
        self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func(new_individual)

def main():
    num_searches = 100
    p1 = 0.5
    p2 = 0.5
    budget = 1000
    dim = 10
    func = lambda x: np.sin(x)
    hbb = EvolutionaryHyperband(budget, dim, num_searches, p1, p2)
    print(f"Hyperband: {hbb.__call__(func)}")
    hbb.budget = 500
    print(f"Hyperband (revised): {hbb.__call__(func)}")

if __name__ == "__main__":
    main()