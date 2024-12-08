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
        def fitness(individual):
            l2 = aoc_logger(self.budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            l2.update(individual)
            return func(individual)

        def tournament_selection(population, func, tournament_size):
            tournament_indices = random.sample(self.population_indices, tournament_size)
            tournament_fitness_values = np.array([fitness(individual) for individual in tournament_indices])
            tournament_parents = []
            for i in range(tournament_size):
                parent_index = tournament_indices[i]
                parent_fitness_value = fitness(individual)
                parent_index = random.choice(self.population_indices)
                parent_fitness_value = fitness(parent_index)
                if parent_fitness_value < parent_fitness_value:
                    parent_index = parent_index
                tournament_parents.append(self.population[parent_index])
            return tournament_parents

        def mutation(individual):
            mutated_individual = individual.copy()
            if random.random() < 0.1:
                mutated_individual[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
            return mutated_individual

        def selection(population, func, tournament_size):
            tournament_parents = tournament_selection(population, func, tournament_size)
            parents = []
            for parent in tournament_parents:
                mutated_parent = mutation(parent)
                parents.append(mutated_parent)
            return parents

        def evolution(population, func, mutation_rate, tournament_size):
            new_population = []
            for _ in range(self.budget):
                parents = selection(population, func, tournament_size)
                for parent in parents:
                    new_population.append(mutation(parent))
            return new_population

        self.population = evolution(self.population, func, 0.1, 3)

    def tournament_selection(self, population, func, tournament_size):
        tournament_indices = random.sample(self.population_indices, tournament_size)
        tournament_fitness_values = np.array([fitness(individual) for individual in tournament_indices])
        tournament_parents = []
        for i in range(tournament_size):
            parent_index = tournament_indices[i]
            parent_fitness_value = fitness(individual)
            parent_index = random.choice(self.population_indices)
            parent_fitness_value = fitness(parent_index)
            if parent_fitness_value < parent_fitness_value:
                parent_index = parent_index
            tournament_parents.append(self.population[parent_index])
        return tournament_parents

    def mutation(self, individual):
        mutated_individual = individual.copy()
        if random.random() < 0.1:
            mutated_individual[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
        return mutated_individual

# Description: Adaptive Black Box Optimization using Metaheuristic Evolutionary Strategies
# Code: 