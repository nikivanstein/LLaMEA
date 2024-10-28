```python
import random
import numpy as np
import copy

class AdaptiveBBOOPIterator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, dim)
        self.population = [self.search_space] * dim
        self.fitness_values = np.zeros(dim)
        self.population_indices = list(range(dim))
        self.new_individuals = []
        self.tournament_size = 3
        self.mutation_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents using tournament selection
            tournament_indices = random.sample(self.population_indices, self.tournament_size)
            tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
            tournament_parents = []
            for i in range(self.tournament_size):
                parent_index = tournament_indices[i]
                parent_fitness_value = tournament_fitness_values[i]
                parent_index = random.choice(self.population_indices)
                parent_fitness_value = self.fitness_values[parent_index]
                if parent_fitness_value < parent_fitness_value:
                    parent_index = parent_index
                tournament_parents.append(self.population[parent_index])
            tournament_indices = random.sample(tournament_indices, self.tournament_size)
            tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
            tournament_parents = []
            for i in range(self.tournament_size):
                parent_index = tournament_indices[i]
                parent_fitness_value = tournament_fitness_values[i]
                parent_index = random.choice(self.population_indices)
                parent_fitness_value = self.fitness_values[parent_index]
                if parent_fitness_value < parent_fitness_value:
                    parent_index = parent_index
                tournament_parents.append(self.population[parent_index])

            # Evolve the population using mutation and selection
            for parent in tournament_parents:
                # Randomly mutate the parent
                mutated_parent = copy.deepcopy(parent)
                if random.random() < self.mutation_rate:
                    mutated_parent[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
                # Select the best parent based on fitness
                best_parent_index = np.argmax(self.fitness_values)
                self.population[self.population_indices[best_parent_index]] = mutated_parent
                self.fitness_values[best_parent_index] = func(mutated_parent)

            # Evaluate the fitness of each individual
            self.evaluate_fitness()

            # Select the best individual
            new_individual = copy.deepcopy(self.population[np.argmax(self.fitness_values)])
            self.new_individuals.append(new_individual)

            # Select parents using tournament selection
            tournament_indices = random.sample(self.population_indices, self.tournament_size)
            tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
            tournament_parents = []
            for i in range(self.tournament_size):
                parent_index = tournament_indices[i]
                parent_fitness_value = tournament_fitness_values[i]
                parent_index = random.choice(self.population_indices)
                parent_fitness_value = self.fitness_values[parent_index]
                if parent_fitness_value < parent_fitness_value:
                    parent_index = parent_index
                tournament_parents.append(self.population[parent_index])
            tournament_indices = random.sample(tournament_indices, self.tournament_size)
            tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
            tournament_parents = []
            for i in range(self.tournament_size):
                parent_index = tournament_indices[i]
                parent_fitness_value = tournament_fitness_values[i]
                parent_index = random.choice(self.population_indices)
                parent_fitness_value = self.fitness_values[parent_index]
                if parent_fitness_value < parent_fitness_value:
                    parent_index = parent_index
                tournament_parents.append(self.population[parent_index])

            # Update the population
            for i in range(self.dim):
                self.population[i] = tournament_parents[i]

            # Update the fitness values
            for i in range(self.dim):
                self.fitness_values[i] = func(self.population[i])

        # Return the best individual
        return self.get_best_individual()

    def select_parents(self):
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
        return tournament_parents

    def evaluate_fitness(self):
        # Evaluate the fitness of each individual
        for i in range(self.dim):
            self.fitness_values[i] = func(self.population[i])

    def get_best_individual(self):
        # Return the best individual
        return self.population[np.argmax(self.fitness_values)]

# Description: Adaptive Black Box Optimization using Metaheuristic Evolutionary Strategies
# Code: 
# 