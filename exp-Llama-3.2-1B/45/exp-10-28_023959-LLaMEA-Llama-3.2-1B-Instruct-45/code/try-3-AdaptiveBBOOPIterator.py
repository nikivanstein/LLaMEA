import random
import numpy as np
import copy
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
        for _ in range(self.budget):
            # Select parents using tournament selection
            tournament_indices = random.sample(self.population_indices, 3)
            tournament_parents = [copy.deepcopy(self.population[i]) for i in tournament_indices]

            # Evolve the population using mutation and selection
            for parent in tournament_parents:
                # Randomly mutate the parent
                mutated_parent = parent.copy()
                if random.random() < 0.1:
                    mutated_parent[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
                # Select the best parent based on fitness
                best_parent_index = np.argmax(self.fitness_values)
                self.population[self.population_indices[best_parent_index]] = mutated_parent
                self.fitness_values[best_parent_index] = func(mutated_parent)

            # Evaluate the fitness of each individual
            for i in range(self.dim):
                self.fitness_values[i] = func(self.population[i])

            # Select the next population using tournament selection
            tournament_indices = random.sample(self.population_indices, 3)
            tournament_parents = [copy.deepcopy(self.population[i]) for i in tournament_indices]
            tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
            tournament_parents = [self.population[parent_index] for parent_index in tournament_parents]
            tournament_parents = [copy.deepcopy(parent) for parent in tournament_parents]
            tournament_fitness_values = np.array([self.fitness_values[parent_index] for parent_index in tournament_indices])
            tournament_parents = [self.population[parent_index] for parent_index in tournament_indices]
            tournament_parents = [copy.deepcopy(parent) for parent in tournament_parents]
            tournament_fitness_values = np.array([self.fitness_values[parent_index] for parent_index in tournament_indices])
            tournament_parents = [self.population[parent_index] for parent_index in tournament_indices]
            tournament_parents = [copy.deepcopy(parent) for parent in tournament_parents]
            tournament_fitness_values = np.array([self.fitness_values[parent_index] for parent_index in tournament_indices])
            tournament_parents = [self.population[parent_index] for parent_index in tournament_indices]
            tournament_parents = [copy.deepcopy(parent) for parent in tournament_parents]
            tournament_fitness_values = np.array([self.fitness_values[parent_index] for parent_index in tournament_indices])
            tournament_parents = [self.population[parent_index] for parent_index in tournament_indices]
            tournament_parents = [copy.deepcopy(parent) for parent in tournament_parents]

            tournament_parents = [self.select_parents(tournament_parents, tournament_fitness_values, tournament_indices, tournament_parents, tournament_fitness_values, tournament_indices, tournament_parents, tournament_fitness_values)]
            tournament_parents = [self.evolve_population(tournament_parents, func)]

            # Update the population
            self.population = tournament_parents

            # Evaluate the fitness of each individual
            for i in range(self.dim):
                self.fitness_values[i] = func(self.population[i])

            # Select the best individual
            best_individual = self.get_best_individual()
            self.population_indices.append(best_individual)

            # Check if the best individual has a fitness value greater than 0
            if self.fitness_values[np.argmax(self.fitness_values)] > 0:
                break

        # Return the best individual
        return self.population[np.argmax(self.fitness_values)]

    def select_parents(self, tournament_parents, tournament_fitness_values, tournament_indices, tournament_parents, tournament_fitness_values, tournament_indices, tournament_parents, tournament_fitness_values):
        # Select the next population using tournament selection
        tournament_indices = random.sample(self.population_indices, 3)
        tournament_parents = [copy.deepcopy(self.population[i]) for i in tournament_indices]
        tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
        tournament_parents = [self.population[parent_index] for parent_index in tournament_parents]
        tournament_parents = [copy.deepcopy(parent) for parent in tournament_parents]
        tournament_fitness_values = np.array([self.fitness_values[parent_index] for parent_index in tournament_indices])
        tournament_parents = [self.population[parent_index] for parent_index in tournament_indices]
        tournament_parents = [copy.deepcopy(parent) for parent in tournament_parents]
        tournament_fitness_values = np.array([self.fitness_values[parent_index] for parent_index in tournament_indices])
        tournament_parents = [self.population[parent_index] for parent_index in tournament_indices]
        tournament_parents = [copy.deepcopy(parent) for parent in tournament_parents]

        # Refine the strategy
        tournament_parents = [self.select_parents(tournament_parents, tournament_fitness_values, tournament_indices, tournament_parents, tournament_fitness_values, tournament_indices, tournament_parents, tournament_fitness_values)]
        tournament_parents = [self.evolve_population(tournament_parents, func)]

        # Update the population
        self.population = tournament_parents

        # Evaluate the fitness of each individual
        for i in range(self.dim):
            self.fitness_values[i] = func(self.population[i])

        # Select the best individual
        best_individual = self.get_best_individual()
        self.population_indices.append(best_individual)

        # Check if the best individual has a fitness value greater than 0
        if self.fitness_values[np.argmax(self.fitness_values)] > 0:
            break

        return tournament_parents

    def evolve_population(self, parents, func):
        # Evolve the population using mutation and selection
        for parent in parents:
            # Randomly mutate the parent
            mutated_parent = parent.copy()
            if random.random() < 0.1:
                mutated_parent[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
            # Select the best parent based on fitness
            best_parent_index = np.argmax(self.fitness_values)
            self.population[self.population_indices[best_parent_index]] = mutated_parent
            self.fitness_values[best_parent_index] = func(mutated_parent)

        return parents

    def evaluate_fitness(self):
        # Evaluate the fitness of each individual
        for i in range(self.dim):
            self.fitness_values[i] = func(self.population[i])

        return self.fitness_values

    def get_best_individual(self):
        # Return the best individual
        return self.population[np.argmax(self.fitness_values)]

# Description: Adaptive Black Box Optimization using Metaheuristic Evolutionary Strategies
# Code: 