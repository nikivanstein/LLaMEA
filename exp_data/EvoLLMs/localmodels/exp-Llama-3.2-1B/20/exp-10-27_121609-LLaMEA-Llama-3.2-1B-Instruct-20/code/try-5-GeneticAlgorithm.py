import numpy as np
import random

class GeneticAlgorithm:
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

        def mutate(individual, mutation_rate):
            if random.random() < mutation_rate:
                index = random.randint(0, self.dim - 1)
                individual[index] = random.uniform(bounds[individual[index]].min(), bounds[individual[index]].max())
            return individual

        def crossover(parent1, parent2):
            if random.random() < 0.5:
                index = random.randint(0, self.dim - 1)
                parent1[index], parent2[index] = parent2[index], parent1[index]
            return parent1

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        # Select the fittest individuals
        self.population = self.select_fittest(self.population, self.population_size, self.budget)

        # Mutate the selected individuals
        self.population = self.mutate(self.population, 0.1)

        # Crossover the selected individuals
        self.population = self.crossover(self.population, self.population_size)

        return self.fitnesses

    def select_fittest(self, population, population_size, budget):
        # Select the fittest individuals using tournament selection
        winners = []
        for _ in range(budget):
            winner = np.random.choice(population, 1)[0]
            winners.append(winner)
            for i in range(population_size):
                if winner!= i:
                    winner = np.random.choice(population, 1)[0]
                    winners.append(winner)
        return np.array(winners)

    def mutate(self, population, mutation_rate):
        return population

    def crossover(self, parent1, parent2):
        child = np.copy(parent1)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = random.uniform(bounds[parent1[i]].min(), bounds[parent1[i]].max())
            if random.random() < 0.5:
                child[i] = random.uniform(bounds[parent2[i]].min(), bounds[parent2[i]].max())
        return child

# Description: Genetic Algorithm for Black Box Optimization
# Code: 