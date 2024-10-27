# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = -np.inf

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if func_value >= self.best_fitness:
                self.best_individual = self.search_space
                self.best_fitness = func_value
        return func_value

def __hebbbo(budget, dim, num_iterations):
    algorithm = HEBBO(budget, dim)
    for _ in range(num_iterations):
        new_individual = algorithm.__call__(np.random.uniform(-5.0, 5.0, dim))
        fitness = algorithm(new_individual)
        if fitness > algorithm.best_fitness:
            algorithm.best_individual = new_individual
            algorithm.best_fitness = fitness
    return algorithm

def __hebbbo_simplified(budget, dim, num_iterations):
    algorithm = HEBBO(budget, dim)
    for _ in range(num_iterations):
        new_individual = algorithm.__call__(np.random.uniform(-5.0, 5.0, dim))
        fitness = algorithm(new_individual)
        if fitness > algorithm.best_fitness:
            algorithm.best_individual = new_individual
            algorithm.best_fitness = fitness
        else:
            # Refine the strategy by changing a random line
            new_individual[0] = np.random.uniform(0, 1)
            fitness = algorithm(new_individual)
            if fitness > algorithm.best_fitness:
                algorithm.best_individual = new_individual
                algorithm.best_fitness = fitness
    return algorithm

# Evaluate the selected solution
def evaluate_bbob(func, budget, dim, num_iterations):
    algorithm = HEBBO(budget, dim)
    for _ in range(num_iterations):
        new_individual = algorithm.__call__(func)
        fitness = algorithm(new_individual)
        if fitness > algorithm.best_fitness:
            algorithm.best_individual = new_individual
            algorithm.best_fitness = fitness
    return algorithm, fitness

# Test the selected solution
func = lambda x: x**2
budget = 1000
dim = 5
num_iterations = 100
algorithm, fitness = evaluate_bbob(func, budget, dim, num_iterations)

# Print the selected solution
print("Selected Solution:")
print("Individual:", algorithm.best_individual)
print("Fitness:", fitness)