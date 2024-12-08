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
        self.population = tournament_parents

        # Evolve the population using mutation and selection
        for _ in range(self.budget):
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
            self.population = tournament_parents

            # Evolve the population using mutation and selection
            self.evolve_population(self.population, func)

            # Evaluate the fitness of each individual
            self.evaluate_fitness()

        # Return the best individual
        return self.get_best_individual()

    def __str__(self):
        return "AdaptiveBBOOPIterator: Black Box Optimization with Metaheuristic Evolutionary Strategies"

    def mutate(self, individual):
        # Randomly mutate the individual
        if random.random() < 0.1:
            index = random.randint(0, self.dim - 1)
            self.population[index] += random.uniform(-1.0, 1.0)
        return individual

    def evolve_population(self, population, func):
        # Evolve the population using mutation and selection
        for individual in population:
            if random.random() < 0.5:
                # Mutate the individual
                individual = self.mutate(individual)
            # Select the best individual based on fitness
            best_individual_index = np.argmax(func(self.population))
            # Replace the individual with the best individual
            self.population[best_individual_index] = individual