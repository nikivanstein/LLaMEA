# Description: Evolutionary algorithm for black box optimization
# Code: 
import random
import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.population = []
        self.fitness_scores = {}
        self.algorithms = {}

    def __call__(self, func):
        # Initialize the population with random individuals
        for _ in range(self.population_size):
            individual = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, self.dim))
            self.population.append(individual)

        # Evaluate the function 1 time
        fitness_scores = {}
        for func in self.fitness_scores:
            fitness_scores[func] = func(self.func_values[func.__name__])

        # Select the fittest individuals
        fittest_individuals = sorted(self.population, key=lambda individual: fitness_scores[individual], reverse=True)[:self.population_size // 2]

        # Select new individuals using evolutionary strategies
        for _ in range(self.budget):
            new_individuals = []
            for individual in fittest_individuals:
                # Select the fittest individual in the current population
                parent1 = random.choice(fittest_individuals)
                parent2 = random.choice(fittest_individuals)

                # Select the crossover point
                crossover_point = random.randint(1, self.dim)

                # Perform crossover
                child1 = np.copy(individual)
                child1[:crossover_point] = parent1[:crossover_point]
                child1[crossover_point:] = parent2[crossover_point:]

                # Select the mutation point
                mutation_point = random.randint(1, self.dim)

                # Perform mutation
                if random.random() < 0.5:
                    child1[mutation_point] += 0.1 * (child1[mutation_point] - child1[mutation_point] * (func(child1) - func(child1[mutation_point])) / (child1[mutation_point] - child1[mutation_point] ** 2))

                new_individuals.append(child1)

            # Replace the fittest individuals with the new ones
            fittest_individuals = sorted(self.population, key=lambda individual: fitness_scores[individual], reverse=True)[:self.population_size // 2]
            self.population = new_individuals + fittest_individuals

        # Evaluate the function 1 time
        fitness_scores = {}
        for func in self.fitness_scores:
            fitness_scores[func] = func(self.func_values[func.__name__])

        # Select the fittest individual
        fittest_individual = sorted(self.population, key=lambda individual: fitness_scores[individual], reverse=True)[0]

        # Return the fittest individual
        return fittest_individual
