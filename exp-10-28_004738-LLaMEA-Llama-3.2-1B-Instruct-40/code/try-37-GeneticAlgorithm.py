import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_population()
        self.fitnesses = np.zeros((len(self.population), self.dim))

    def generate_population(self):
        population = []
        for _ in range(100):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def evaluate_fitness(self, individual, func):
        fitness = func(individual)
        self.fitnesses[self.population.index(individual)] = fitness
        return fitness

    def __call__(self, func, bounds):
        population = self.population
        for _ in range(self.budget):
            for individual in population:
                fitness = self.evaluate_fitness(individual, func)
                if fitness < 0:
                    individual = [x - 1 for x in individual]
                elif fitness > 0:
                    individual = [x + 1 for x in individual]
            population = sorted(population, key=lambda x: self.fitnesses[x], reverse=True)
            individual = population[0]
            if random.random() < 0.4:
                individual = random.uniform(bounds[0], bounds[1])
            elif random.random() < 0.8:
                individual = random.uniform(bounds[0], bounds[1])
            elif random.random() < 0.95:
                individual = random.uniform(bounds[0], bounds[1])
        return individual

# Description: Evolutionary Optimization using Genetic Algorithm
# Code: 