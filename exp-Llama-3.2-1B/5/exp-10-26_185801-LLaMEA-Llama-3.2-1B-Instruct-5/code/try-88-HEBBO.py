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

class HBGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if random.random() < 0.05:
                new_individual = self.evaluate_fitness(random.uniform(self.search_space))
            else:
                new_individual = self.optimize(func, self.search_space)
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

def evaluateBBOB(func, search_space, budget):
    individual = random.uniform(search_space)
    for _ in range(budget):
        func_value = func(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
    return func_value

def optimize(func, search_space):
    # Simple mutation strategy: swap two random elements
    index1, index2 = random.sample(range(len(search_space)), 2)
    search_space[index1], search_space[index2] = search_space[index2], search_space[index1]
    return func(search_space)

# Initialize HEBBO with a budget of 1000 evaluations
hebbbo = HEBBO(1000, 10)

# Evaluate the BBOB function
func = lambda x: np.sin(x)
bboo_value = evaluateBBOB(func, [-10, 10], 1000)

# Update the HEBBO algorithm
hebbbo.search_space = np.linspace(-10, 10, 20)

# Print the updated HEBBO algorithm
print("Updated HEBBO algorithm:")
print(f"Search space: {hebbbo.search_space}")
print(f"Fitness: {evaluateBBOB(func, hebbbo.search_space, 1000)}")