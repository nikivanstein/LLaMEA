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
        self.search_space = [x for x in self.search_space if x not in best_func]

        # Perform mutation and crossover
        mutated_search_space = self.search_space.copy()
        for _ in range(5):  # Perform 5 mutations
            mutation_index = random.randint(0, len(mutated_search_space) - 1)
            mutated_search_space[mutation_index] = random.uniform(-5.0, 5.0)

        # Perform selection
        selected_individual = random.sample(mutated_search_space, 1)[0]

        # Evaluate the new individual
        new_individual = self.evaluate_fitness(selected_individual)

        return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the function using the budget
        num_evals = min(self.budget, len(individual))
        func_values = [individual[i] for i in range(num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        new_search_space = [x for x in individual if x not in best_func]

        return best_func, new_search_space