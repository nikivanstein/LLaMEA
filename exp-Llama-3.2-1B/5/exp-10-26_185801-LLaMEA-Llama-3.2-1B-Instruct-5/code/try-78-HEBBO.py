import numpy as np
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

    def mutate(self, individual):
        # Refine the strategy by changing the individual lines
        # to refine its approach
        new_individual = individual.copy()
        if np.random.rand() < 0.1:  # 10% chance of mutation
            if np.random.rand() < 0.5:  # 50% chance of changing the first element
                new_individual[0] = np.random.uniform(-5.0, 5.0)
            else:
                new_individual[1] = np.random.uniform(-5.0, 5.0)
        return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual using the HEBBO algorithm
        func_value = self.__call__(individual)
        return func_value

# Initialize the HEBBO algorithm
HEBBO_instance = HEBBO(100, 10)

# Define a test function
def test_function(x):
    return np.sin(x)

# Evaluate the fitness of the individual
individual = np.array([1.0])
fitness = HEBBO_instance.evaluate_fitness(individual)
print(fitness)

# Refine the strategy by changing the individual lines
individual = HEBBO_instance.mutate(individual)

# Evaluate the fitness of the refined individual
fitness = HEBBO_instance.evaluate_fitness(individual)
print(fitness)