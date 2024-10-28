# Description: Adaptive Differential Evolution for Black Box Optimization
# Code: 
# ```python
import numpy as np
from scipy.optimize import differential_evolution
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def __str__(self):
        return f"DABU: {self.dim}D optimization with Area over the convergence curve (AOCC) score of {self.score:.4f}"

def adaptive_differential_evolution(func, bounds, budget):
    # Initialize the population with random solutions
    population = [random.uniform(bounds[0], bounds[1]) for _ in range(50)]

    for _ in range(10):
        # Evaluate the objective function for each individual in the population
        results = differential_evolution(lambda x: -func(x), [(bounds[0], bounds[1])], popcount=population)

        # Update the population with the best individual
        population = [x for x, _ in results]

    # Evaluate the objective function for the final population
    results = differential_evolution(lambda x: -func(x), [(bounds[0], bounds[1])], popcount=population)

    # Return the best solution
    return population[0]

# Define the test function
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

# Run the algorithm
result = adaptive_differential_evolution(test_function, (-10, 10), 1000)

# Print the result
print(f"Result: {result}")