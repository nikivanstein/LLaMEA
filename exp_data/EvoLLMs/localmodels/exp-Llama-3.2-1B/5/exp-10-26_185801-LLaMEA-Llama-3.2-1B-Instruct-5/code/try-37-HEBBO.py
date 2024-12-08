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

class HESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if random.random() < 0.05:
                # Randomly choose a new search direction
                new_direction = np.random.rand(self.dim)
                new_individual = self.evaluate_fitness(func, new_direction)
            else:
                # Refine the individual's strategy
                new_individual = self.fine_tune_individual(func, self.search_space)
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def fine_tune_individual(self, func, search_space):
        # Refine the individual's strategy by changing the search direction
        # with a probability of 0.05
        new_direction = np.random.rand(self.dim)
        new_individual = func(search_space + new_direction)
        return new_individual

class HESO_BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if random.random() < 0.05:
                # Randomly choose a new search direction
                new_direction = np.random.rand(self.dim)
                new_individual = self.evaluate_fitness(func, new_direction)
            else:
                # Refine the individual's strategy by changing the search direction
                # with a probability of 0.05
                new_direction = np.random.rand(self.dim)
                new_individual = self.fine_tune_individual(func, new_direction)
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

def evaluate_bbob(func, search_space, budget):
    # Evaluate the black box function for the given search space
    # and budget
    return func(search_space)

def evaluate_fitness(individual, search_space):
    # Evaluate the fitness of the individual
    func_value = evaluate_bbob(lambda x: individual(x), search_space, 1000)
    return func_value

def fine_tune_individual(individual, search_space):
    # Refine the individual's strategy by changing the search direction
    # with a probability of 0.05
    new_direction = np.random.rand(len(individual))
    new_individual = individual(new_direction)
    return new_individual

# Create an instance of HESO
heso = HESO(1000, 5)

# Evaluate the BBOB test suite
for func in ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24"]:
    print(f"Evaluating {func}...")
    func_value = evaluate_bbob(lambda x: heso(individual=x), heso.search_space, 1000)
    print(f"Fitness: {func_value}")
    print()