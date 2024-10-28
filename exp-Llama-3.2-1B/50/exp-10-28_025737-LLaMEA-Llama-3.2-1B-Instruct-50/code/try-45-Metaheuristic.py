# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

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
        new_individual = self.search_space[:]

        # Refine the strategy
        if len(new_individual) < 2:
            new_individual = self.better_strategy(func, new_individual, self.budget)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in new_individual]

        # Return the best function value
        return best_func

    def better_strategy(self, func, individual, budget):
        # Create a copy of the individual
        new_individual = individual.copy()

        # Evaluate the function for the new individual
        num_evals = min(self.budget, len(func(new_individual)))

        # Select the best function value
        best_func = max(set(func(new_individual)), key=func(new_individual).count)

        # Update the search space
        new_individual = [x for x in new_individual if x not in best_func]

        # Refine the strategy
        if len(new_individual) < 2:
            new_individual = self.better_strategy(func, new_individual, self.budget)

        # Update the search space
        self.search_space = [x for x in new_individual if x not in new_individual]

        return new_individual

# Test the algorithm
def func(x):
    return np.sin(x) + 2 * np.cos(x) + 3 * np.exp(x)

metaheuristic = Metaheuristic(100, 10)
print(metaheuristic(func))