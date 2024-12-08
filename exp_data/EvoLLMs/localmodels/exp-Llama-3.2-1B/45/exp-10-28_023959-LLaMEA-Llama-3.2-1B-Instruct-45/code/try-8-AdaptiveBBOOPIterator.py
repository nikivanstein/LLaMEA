import random
import numpy as np
from collections import deque

class AdaptiveBBOOPIterator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, dim)
        self.population = [self.search_space] * dim
        self.fitness_values = np.zeros(dim)
        self.population_indices = list(range(dim))
        self.population_history = deque(maxlen=self.budget)

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents using tournament selection
            tournament_indices = random.sample(self.population_indices, 3)
            tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
            tournament_parents = []
            for i in range(3):
                parent_index = tournament_indices[i]
                parent_fitness_value = self.fitness_values[parent_index]
                parent_index = random.choice(self.population_indices)
                parent_fitness_value = self.fitness_values[parent_index]
                if parent_fitness_value < parent_fitness_value:
                    parent_index = parent_index
                tournament_parents.append(self.population[parent_index])
            tournament_indices = tournament_indices[:3]
            tournament_parents = tournament_parents[:3]
            self.population_history.extend(tournament_parents)

            # Evolve the population using mutation and selection
            new_individuals = []
            for _ in range(self.dim):
                parent_index = random.choice(tournament_indices)
                parent_fitness_value = self.fitness_values[parent_index]
                parent_index = random.choice(self.population_indices)
                parent_fitness_value = self.fitness_values[parent_index]
                if parent_fitness_value < parent_fitness_value:
                    parent_index = parent_index
                mutated_parent = self.population[parent_index].copy()
                if random.random() < 0.1:
                    mutated_parent[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
                new_individual = mutated_parent.copy()
                self.population_history.append(new_individual)
                self.fitness_values[new_individual.index] = func(new_individual)

            # Select the best individual based on fitness
            new_individuals = [self.population[i] for i in self.population_history]
            self.population_history = deque(new_individuals)

            # Update the best individual
            self.get_best_individual()

        # Return the best individual
        return self.get_best_individual()

    def get_best_individual(self):
        # Return the best individual
        return self.population[np.argmax(self.fitness_values)]

# Description: Adaptive Black Box Optimization using Metaheuristic Evolutionary Strategies
# Code: 