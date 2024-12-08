import random
import numpy as np
from scipy.optimize import minimize

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

    def mutate(self, individual):
        # Refine the strategy by changing 0.45 of the individual's lines
        mutated_individual = individual.copy()
        mutated_individual[0] = random.uniform(0.0, 0.45) + mutated_individual[0]
        mutated_individual[1] = random.uniform(0.0, 0.45) + mutated_individual[1]

        return mutated_individual

# Initialize the selected solution
algorithm = Metaheuristic(100, 10)

# Define the black box function
def func(x):
    return x[0] + x[1] + x[2] + x[3]

# Evaluate the function
best_func = algorithm(func)

# Optimize the function using the selected algorithm
best_individual = algorithm(func)
best_fitness = func(best_individual)

# Print the results
print(f"Best function: {best_func} with fitness: {best_fitness}")
print(f"Best individual: {best_individual}")

# Update the algorithm
algorithm = Metaheuristic(100, 10)