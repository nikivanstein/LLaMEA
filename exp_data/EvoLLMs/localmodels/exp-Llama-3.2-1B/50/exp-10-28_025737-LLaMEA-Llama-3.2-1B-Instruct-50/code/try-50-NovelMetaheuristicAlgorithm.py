import numpy as np
import random

class NovelMetaheuristicAlgorithm:
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
        # Refine the strategy by changing the individual's line
        if random.random() < 0.45:
            # Select a random line from the search space
            line = random.choice(self.search_space)

            # Change the individual's line
            new_individual = [x if x!= line else x + random.uniform(-0.1, 0.1) for x in individual]
        else:
            # Keep the individual's line unchanged
            new_individual = individual

        return new_individual

# Initialize the algorithm with a budget and dimension
budget = 1000
dim = 10
algorithm = NovelMetaheuristicAlgorithm(budget, dim)

# Evaluate the function for the initial population
initial_population = [algorithm.__call__(func) for func in [lambda x: x**2, lambda x: x**3]]
print("Initial population:", initial_population)

# Evaluate the function for the new population
new_population = [algorithm.__call__(func) for func in [lambda x: x**2, lambda x: x**3]]
print("New population:", new_population)

# Refine the search space
algorithm.search_space = [x for x in algorithm.search_space if x not in new_population[0]]
print("Refined search space:", algorithm.search_space)

# Evaluate the function for the new population again
new_population = [algorithm.__call__(func) for func in [lambda x: x**2, lambda x: x**3]]
print("New population:", new_population)

# Refine the search space again
algorithm.search_space = [x for x in algorithm.search_space if x not in new_population[0]]
print("Refined search space:", algorithm.search_space)

# Evaluate the function for the new population again
new_population = [algorithm.__call__(func) for func in [lambda x: x**2, lambda x: x**3]]
print("New population:", new_population)