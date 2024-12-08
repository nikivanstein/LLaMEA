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

        return best_func

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func, iterations=100, mutation_rate=0.01):
        # Initialize the population with random individuals
        population = [self.__call__(func) for _ in range(100)]

        # Evolve the population
        for _ in range(iterations):
            # Select the best individual
            best_individual = max(population, key=self.__call__)

            # Create a new individual by mutation
            new_individual = self.__call__(best_individual)

            # Evaluate the new individual
            new_individual_value = self.__call__(new_individual)

            # Check if the mutation rate is exceeded
            if random.random() < mutation_rate:
                # Randomly select a function value from the search space
                func_value = random.choice(self.search_space)

                # Update the new individual's value
                new_individual_value = self.__call__(func_value)

            # Update the population
            population.append(new_individual)

        # Return the best individual
        return max(population, key=self.__call__)

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 