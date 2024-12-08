import numpy as np
import random

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = np.zeros((self.population_size, self.dim, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if random.random() < 0.2:
                new_dim = random.randint(0, self.dim - 1)
                individual[new_dim] = random.uniform(-5.0, 5.0)
            return individual

        def crossover(parent1, parent2):
            if random.random() < 0.5:
                index = random.randint(0, self.dim - 1)
                parent1[index] = parent2[index]
            return parent1

        def selection(population):
            return np.array([individual for individual in population if individual[0] <= 5.0])

        def mutate_and_crossover(population):
            return [mutate(individual) for individual in population]

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual[0]] + 1e-6:
                self.fitnesses[individual[0]] = fitness
            return individual

        for _ in range(self.budget):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.population[np.random.randint(0, self.population_size)]
                parent2 = self.population[np.random.randint(0, self.population_size)]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                if random.random() < 0.5:
                    child1 = mutate_and_crossover([child1, child2])
                new_population.append(evaluate_fitness(child1))
            self.population = new_population
            self.population_history = np.concatenate((self.population_history, self.population))

        return self.fitnesses

# Description: Evolutionary Optimization Algorithm
# Code: 