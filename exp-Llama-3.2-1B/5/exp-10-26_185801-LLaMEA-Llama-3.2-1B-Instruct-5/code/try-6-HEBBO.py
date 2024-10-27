import numpy as np
import random

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

class HBDFO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if random.random() < 0.05:
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            else:
                best_individual = self.evaluate_fitness(func, self.search_space)
                self.search_space = np.linspace(self.search_space[best_individual], 5.0, self.dim)
        return self.evaluate_fitness(func, self.search_space)

class HBDFO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if random.random() < 0.05:
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            else:
                best_individual = self.evaluate_fitness(func, self.search_space)
                if np.isnan(self.evaluate_fitness(func, self.search_space)):
                    raise ValueError("Invalid function value")
                if np.isinf(self.evaluate_fitness(func, self.search_space)):
                    raise ValueError("Function value must be between 0 and 1")
                self.search_space = np.linspace(self.search_space[best_individual], 5.0, self.dim)
        return self.evaluate_fitness(func, self.search_space)

class HBDFO3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if random.random() < 0.05:
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            else:
                best_individual = self.evaluate_fitness(func, self.search_space)
                if np.isnan(self.evaluate_fitness(func, self.search_space)):
                    raise ValueError("Invalid function value")
                if np.isinf(self.evaluate_fitness(func, self.search_space)):
                    raise ValueError("Function value must be between 0 and 1")
                self.search_space = np.linspace(self.search_space[best_individual], 5.0, self.dim)
        return self.evaluate_fitness(func, self.search_space)

def evaluate_fitness(func, search_space):
    return func(search_space)

def main():
    budget = 1000
    dim = 10
    print("HEBBO: ", HEBBO(budget, dim).__call__(lambda x: x[0]))
    print("HBDFO: ", HBDFO(budget, dim).__call__(lambda x: x[0]))
    print("HBDFO2: ", HBDFO2(budget, dim).__call__(lambda x: x[0]))
    print("HBDFO3: ", HBDFO3(budget, dim).__call__(lambda x: x[0]))

if __name__ == "__main__":
    main()