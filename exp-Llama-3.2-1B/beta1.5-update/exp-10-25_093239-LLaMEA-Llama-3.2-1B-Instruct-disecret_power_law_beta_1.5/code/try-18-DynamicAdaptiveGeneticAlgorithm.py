import random
import math
import numpy as np

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func, population_size=50, max_iter=1000):
        """
        :param func: The black box function to optimize.
        :param population_size: The size of the population. Defaults to 50.
        :param max_iter: The maximum number of iterations. Defaults to 1000.
        :return: The best individual.
        """
        for _ in range(max_iter):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individuals = np.array([self.select_next_individual() for _ in range(population_size)])
            next_individuals = np.random.choice(population_size, size=population_size, replace=True, p=[self.fitnesses[i] / self.fitnesses[-1] for i in range(population_size)])
            next_individuals = np.clip(next_individuals, -5.0, 5.0)
            # Update the fitness and the population
            updated_individuals = np.array([func(individual) for individual in next_individuals])
            updated_fitnesses = np.array([func(individual) for individual in updated_individuals])
            self.fitnesses = updated_fitnesses
            self.population = updated_individuals
        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy by changing the line of code to refine its strategy
        # Use a weighted sum of the fitness and the dimension
        return np.array([x + y * self.dim for x, y in zip(self.population, [0.5, 0.3])])

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.