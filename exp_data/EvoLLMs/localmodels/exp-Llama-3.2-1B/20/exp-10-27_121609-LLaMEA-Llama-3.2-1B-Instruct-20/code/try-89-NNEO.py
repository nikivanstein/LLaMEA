# Black Box Optimization using NNEO Algorithm
# Description: An evolutionary algorithm that optimizes black box functions using a combination of mutation and selection.

import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            return x + np.random.uniform(-1.0, 1.0)

        def select_parents(population):
            population = np.array(population)
            population = population[:self.population_size // 2]
            population = population[np.random.choice(population.shape[0], self.population_size - self.population_size // 2, replace=False)]
            population = np.concatenate((population, population[:self.population_size - self.population_size // 2]))
            return population

        def evaluate_fitness(individual, logger):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual] + 1e-6:
                self.fitnesses[individual] = fitness
                logger.log("Fitness improved: " + str(individual) + " -> " + str(fitness))
            return fitness

        def crossover(parent1, parent2):
            child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
            return child

        def mutate_and_crossover(parent, mutation_rate):
            child = mutate(parent)
            while np.random.rand() < mutation_rate:
                child = mutate(child)
            return crossover(parent, child)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = evaluate_fitness(x, self.logger)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

            new_population = []
            for _ in range(self.population_size):
                parent = np.random.choice(self.population, self.dim)
                parent = select_parents(parent)
                child = mutate_and_crossover(parent, 0.2)
                new_population.append(child)

            self.population = np.array(new_population)
            self.population_history.append(self.population)

            if len(self.population_history) > self.budget:
                self.population = self.population_history[-1]

        return self.fitnesses

# Test the algorithm
def test_nneo():
    func = lambda x: x**2
    nneo = NNEO(100, 10)
    nneo.initialize_single(func)
    nneo.optimize(func)

# Run the test
test_nneo()