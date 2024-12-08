import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

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

        return self.fitnesses

    def mutate(self, individual):
        bounds = bounds(individual)
        if random.random() < 0.2:
            new_individual = individual.copy()
            new_individual[random.randint(0, self.dim-1)] = random.uniform(bounds[0], bounds[1])
            return new_individual
        else:
            return individual

    def crossover(self, parent1, parent2):
        bounds = bounds(parent1)
        if random.random() < 0.2:
            child1 = parent1.copy()
            child2 = parent2.copy()
            child1[random.randint(0, self.dim-1)] = parent2[random.randint(0, self.dim-1)]
            return child1, child2
        else:
            return parent1, parent2

    def selection(self):
        parents = np.random.choice(self.population, self.population_size, replace=False)
        return parents

    def evaluate_fitness(self, func, population):
        def objective(x):
            return func(x)

        for individual in population:
            fitness = objective(individual)
            self.fitnesses[population.index(individual)] = fitness

        return self.fitnesses

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using mutation, crossover, and selection.