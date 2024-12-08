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
        for _ in range(10):
            # Refine the strategy
            new_individual = self.refine_strategy(new_individual, func_values, best_func)

        # Update the search space
        self.search_space = new_individual

        return best_func

    def refine_strategy(self, individual, func_values, best_func):
        # Initialize the new individual
        new_individual = individual.copy()

        # Select the next individual
        for _ in range(5):
            # Select the best individual based on the probability 0.45
            best_individual = max(set(func_values), key=func_values.count)

            # Select the next individual based on the probability 0.55
            new_individual = self.select_next_individual(new_individual, best_individual)

        return new_individual

    def select_next_individual(self, individual, best_individual):
        # Select the next individual based on the probability 0.45
        selection_prob = 0.45
        selection_prob = selection_prob / (selection_prob + 0.55)
        probabilities = [selection_prob, 1 - selection_prob]

        # Select the next individual based on the probabilities
        new_individual = random.choices(individual, weights=probabilities, k=1)[0]

        return new_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 