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
        # Select parents using tournament selection with probability 0.45
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
        tournament_parents = np.array(tournament_parents)

        # Evolve the population using mutation and selection
        for _ in range(self.budget):
            # Select parents using tournament selection with probability 0.45
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
            tournament_parents = np.array(tournament_parents)

            # Evolve the population using mutation and selection
            mutated_parents = []
            for parent in tournament_parents:
                # Randomly mutate the parent
                mutated_parent = parent.copy()
                if random.random() < 0.1:
                    mutated_parent[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
                # Select the best parent based on fitness
                best_parent_index = np.argmax(self.fitness_values)
                mutated_parent[self.population_indices[best_parent_index]] = parent
                self.fitness_values[best_parent_index] = func(mutated_parent)
                mutated_parents.append(mutated_parent)

            # Select parents using tournament selection with probability 0.45
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
            tournament_parents = np.array(tournament_parents)

            # Evolve the population using mutation and selection
            mutated_parents = []
            for parent in tournament_parents:
                # Randomly mutate the parent
                mutated_parent = parent.copy()
                if random.random() < 0.1:
                    mutated_parent[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
                # Select the best parent based on fitness
                best_parent_index = np.argmax(self.fitness_values)
                mutated_parent[self.population_indices[best_parent_index]] = parent
                self.fitness_values[best_parent_index] = func(mutated_parent)
                mutated_parents.append(mutated_parent)

            # Select the best parent based on fitness
            best_parent_index = np.argmax(self.fitness_values)
            self.population[self.population_indices[best_parent_index]] = mutated_parents[best_parent_index]
            self.fitness_values[best_parent_index] = func(mutated_parents[best_parent_index])

        # Return the best individual
        return self.get_best_individual()

    def select_parents(self):
        # Select parents using tournament selection with probability 0.45
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

# Description: Adaptive Black Box Optimization using Metaheuristic Evolutionary Strategies
# Code: 