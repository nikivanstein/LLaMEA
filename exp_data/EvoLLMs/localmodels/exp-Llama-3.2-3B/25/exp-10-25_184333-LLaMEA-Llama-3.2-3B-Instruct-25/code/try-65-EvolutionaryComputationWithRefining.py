import numpy as np
import random

class EvolutionaryComputationWithRefining:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.swarm_size = 10
        self.warmup = 10
        self.c1 = 1.5
        self.c2 = 1.5
        self.rho = 0.99
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        for individual in self.population:
            func(individual)

    def update(self, prob_refine=0.25):
        for _ in range(self.swarm_size):
            for individual in self.population:
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                v1 = r1 * self.c1 * (individual - self.population[random.randint(0, self.population_size - 1)])
                v2 = r2 * self.c2 * (self.population[random.randint(0, self.population_size - 1)] - individual)
                individual += v1 + v2
                if individual[0] < self.lower_bound:
                    individual[0] = self.lower_bound
                if individual[0] > self.upper_bound:
                    individual[0] = self.upper_bound
                if individual[1] < self.lower_bound:
                    individual[1] = self.lower_bound
                if individual[1] > self.upper_bound:
                    individual[1] = self.upper_bound
                if random.random() < prob_refine:
                    individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        return self.population

    def refine(self, func):
        for _ in range(self.budget - self.warmup):
            self.evaluate(func)
            self.population = self.update()
            min_fitness = np.inf
            min_individual = None
            for individual in self.population:
                if func(individual) < min_fitness:
                    min_fitness = func(individual)
                    min_individual = individual
            self.population = [min_individual]
        return self.population[0]

# Example usage:
def func(x):
    return np.sum(x**2)

ecwr = EvolutionaryComputationWithRefining(100, 10)
result = ecwr.refine(func)
print(result)