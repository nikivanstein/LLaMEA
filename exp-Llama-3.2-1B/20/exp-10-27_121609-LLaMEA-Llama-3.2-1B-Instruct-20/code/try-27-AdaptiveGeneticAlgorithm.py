import numpy as np
import random

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        # Select the best individual based on the fitness
        best_individual = np.argmax(self.fitnesses)

        # Initialize the new population
        new_population = np.copy(self.population)

        # Perform mutation on the selected individual
        if random.random() < self.mutation_rate:
            new_individual = self.evaluate_fitness(new_individual)
            if new_individual is not None:
                new_individual = self.evaluate_fitness(new_individual)
                if new_individual is not None:
                    new_individual = self.evaluate_fitness(new_individual)
                    if new_individual is not None:
                        new_individual = self.evaluate_fitness(new_individual)

        # Refine the new population based on the probability of mutation
        if random.random() < 0.2:
            mutation_indices = np.random.choice(self.population_size, self.population_size, replace=False)
            for i in mutation_indices:
                new_individual = self.evaluate_fitness(new_population[i])
                if new_individual is not None:
                    new_individual = self.evaluate_fitness(new_individual)
                    if new_individual is not None:
                        new_individual = self.evaluate_fitness(new_individual)

        # Replace the old population with the new population
        self.population = np.concatenate((self.population, new_population))

        # Evaluate the new population
        self.fitnesses = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            x = self.population[i]
            fitness = objective(x)
            if fitness < self.fitnesses[i, x] + 1e-6:
                self.fitnesses[i, x] = fitness

        return self.fitnesses

# One-line description: 
# Novel black box optimization algorithm using adaptive genetic algorithm.
# 
# The algorithm selects the best individual based on the fitness, then initializes a new population based on the selected individual.
# The new population is refined based on a probability of mutation, and the process is repeated until the budget is exhausted.