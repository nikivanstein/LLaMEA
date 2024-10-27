import numpy as np
import random
import copy

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        population_size = 50
        probability_refine = 0.4

        # Initialize population
        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(population_size)]

        # Refine strategy for 20% of the population
        for _ in range(int(population_size * probability_refine)):
            i = random.randint(0, population_size - 1)
            new_individual = copy.deepcopy(population[i])
            new_individual = self.refine_strategy(new_individual, func, self.bounds)
            population[i] = new_individual

        # Evaluate fitness
        fitness = [self.evaluate_fitness(individual, func, self.bounds) for individual in population]

        # Select fittest individuals
        fittest_individuals = sorted(zip(population, fitness), key=lambda x: x[1])[:10]

        # Perform crossover and mutation
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = self.crossover(parent1[0], parent2[0])
            child = self.mutate(child, func, self.bounds)
            new_population.append(child)

        return self.evaluate_fitness(new_population, func, self.bounds)

    def refine_strategy(self, individual, func, bounds):
        # Refine strategy by changing individual lines with probability 0.4
        new_individual = copy.deepcopy(individual)
        for i in range(self.dim):
            if random.random() < 0.4:
                new_individual[i] = random.uniform(bounds[i][0], bounds[i][1])
        return new_individual

    def evaluate_fitness(self, individual, func, bounds):
        return func(individual)

    def crossover(self, parent1, parent2):
        # Perform crossover using uniform crossover
        child = []
        for i in range(self.dim):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def mutate(self, individual, func, bounds):
        # Perform mutation using Gaussian mutation
        for i in range(self.dim):
            if random.random() < 0.1:
                individual[i] += np.random.normal(0, 1)
                individual[i] = max(bounds[i][0], min(bounds[i][1], individual[i]))
        return individual