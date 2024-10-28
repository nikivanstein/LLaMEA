import random
import numpy as np

class AdaptiveBBOOPIterator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, dim)
        self.population = [self.search_space] * dim
        self.fitness_values = np.zeros(dim)
        self.population_indices = list(range(dim))

    def __call__(self, func):
        def _optimize(func, budget, dim):
            while True:
                # Select parents using tournament selection
                tournament_indices = random.sample(self.population_indices, budget)
                tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
                tournament_parents = []
                for i in range(budget):
                    parent_index = tournament_indices[i]
                    parent_fitness_value = self.fitness_values[parent_index]
                    if parent_fitness_value < self.fitness_values[parent_index]:
                        parent_index = parent_index
                    tournament_parents.append(self.population[parent_index])
                # Evolve the population using mutation and selection
                self.evolve_population(tournament_parents, func)
                # Evaluate the fitness of each individual
                self.evaluate_fitness()
                # Update the population with the best individuals
                self.population = [func(individual) for individual in self.population]
                # Check if the population is within the search space
                if np.all(self.fitness_values >= -5.0) and np.all(self.fitness_values <= 5.0):
                    return self.population
        return _optimize

# Code: 