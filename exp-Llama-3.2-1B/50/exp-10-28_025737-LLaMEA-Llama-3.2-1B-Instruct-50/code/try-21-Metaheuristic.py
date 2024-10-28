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
        new_individual = self.search_space.copy()
        for _ in range(2):  # Refine the strategy
            new_individual = self.refine_strategy(new_individual, best_func, func_values)

        return best_func, new_individual

    def refine_strategy(self, individual, best_func, func_values):
        # Select the next individual based on the probability of change
        change_prob = 0.45
        change_individual = individual.copy()
        change_probabilities = np.random.rand(len(individual))
        change_indices = np.random.choice(len(individual), size=len(individual), p=change_probabilities)
        for index in change_indices:
            change_individual[index], best_func = best_func, func_values[index]

        # Select the best individual based on the probability of change
        change_prob = 0.45
        change_probabilities = np.random.rand(len(individual))
        change_indices = np.random.choice(len(individual), size=len(individual), p=change_probabilities)
        for index in change_indices:
            new_individual[index], best_func = best_func, func_values[index]

        return new_individual

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Selects the next individual based on a probability of change and refinement strategy
# to optimize the function value.