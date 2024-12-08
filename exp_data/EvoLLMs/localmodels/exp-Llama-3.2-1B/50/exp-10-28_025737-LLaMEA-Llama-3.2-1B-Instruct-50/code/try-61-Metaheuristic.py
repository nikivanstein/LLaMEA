import numpy as np
import random

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

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.metaheuristic = Metaheuristic(budget, dim)

    def __call__(self, func):
        # Refine the search space based on the probability of mutation
        probability = 0.45
        mutation_probabilities = [0.5] * self.metaheuristic.dim
        mutation_probabilities[random.randint(0, self.metaheuristic.dim - 1)] = 0.0

        for _ in range(self.metaheuristic.budget):
            new_individual = self.metaheuristic.evaluate_fitness(np.random.choice(self.metaheuristic.search_space, self.metaheuristic.dim))

            # Evaluate the new individual
            func_values = [func(x) for x in new_individual]

            # Select the best function value
            best_func = max(set(func_values), key=func_values.count)

            # Update the search space
            self.metaheuristic.search_space = [x for x in self.metaheuristic.search_space if x not in best_func]

            # Mutate the search space
            mutation_probabilities[random.randint(0, self.metaheuristic.dim - 1)] += probability

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 