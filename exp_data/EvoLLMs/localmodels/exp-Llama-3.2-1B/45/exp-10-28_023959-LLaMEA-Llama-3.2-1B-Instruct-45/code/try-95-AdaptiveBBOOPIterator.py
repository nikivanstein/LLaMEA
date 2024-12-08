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

    def __call__(self, func):
        # Initialize the population with random solutions
        self.population = [self.search_space] * self.dim
        self.fitness_values = np.zeros(self.dim)

        # Select parents using tournament selection
        tournament_size = 3
        tournament_indices = random.sample(self.population_indices, tournament_size)
        tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
        tournament_parents = []
        for i in range(tournament_size):
            parent_index = tournament_indices[i]
            parent_fitness_value = tournament_fitness_values[i]
            parent_index = random.choice(self.population_indices)
            parent_fitness_value = self.fitness_values[parent_index]
            if parent_fitness_value < parent_fitness_value:
                parent_index = parent_index
            tournament_parents.append(self.population[parent_index])

        # Evolve the population using mutation and selection
        while self.budget > 0:
            # Select parents using tournament selection
            tournament_indices = random.sample(self.population_indices, tournament_size)
            tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
            tournament_parents = []
            for i in range(tournament_size):
                parent_index = tournament_indices[i]
                parent_fitness_value = tournament_fitness_values[i]
                parent_index = random.choice(self.population_indices)
                parent_fitness_value = self.fitness_values[parent_index]
                if parent_fitness_value < parent_fitness_value:
                    parent_index = parent_index
                tournament_parents.append(self.population[parent_index])

            # Evolve the population using mutation and selection
            new_individuals = []
            for _ in range(10):  # Number of generations
                # Select parents using tournament selection
                tournament_indices = random.sample(self.population_indices, tournament_size)
                tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
                tournament_parents = []
                for i in range(tournament_size):
                    parent_index = tournament_indices[i]
                    parent_fitness_value = tournament_fitness_values[i]
                    parent_index = random.choice(self.population_indices)
                    parent_fitness_value = self.fitness_values[parent_index]
                    if parent_fitness_value < parent_fitness_value:
                        parent_index = parent_index
                    tournament_parents.append(self.population[parent_index])

                # Evolve the population using mutation and selection
                new_individuals = []
                for individual in self.population:
                    # Randomly mutate the individual
                    mutated_individual = individual.copy()
                    if random.random() < 0.1:
                        mutated_individual[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
                    # Select the best individual based on fitness
                    best_individual_index = np.argmax(self.fitness_values)
                    self.population[self.population_indices[best_individual_index]] = mutated_individual
                    self.fitness_values[best_individual_index] = func(mutated_individual)

                # Update the population
                self.population = new_individuals
                self.fitness_values = np.zeros(len(self.population))

                # Evaluate the fitness of each individual
                for i in range(self.dim):
                    self.fitness_values[i] = func(self.population[i])

                # Check if the population has reached the budget
                if len(self.population) <= self.budget:
                    break

            # Update the population indices
            self.population_indices = list(range(len(self.population)))

            # Update the best individual
            self.population[np.argmax(self.fitness_values)] = self.search_space

            # Update the budget
            self.budget -= 1

        # Return the best individual
        return self.population[np.argmax(self.fitness_values)]

# Description: Adaptive Black Box Optimization using Metaheuristic Evolutionary Strategies
# Code: 