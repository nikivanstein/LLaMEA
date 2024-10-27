import numpy as np
import random

class DiversityDrivenEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.drift_rate = 0.1
        self.population = self.initialize_population()
        self.differential_evolution = DifferentialEvolution(self.population_size, self.dim)
        self.genetic_drift = GeneticDrift(self.population_size, self.dim)

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        self.population = self.select_parents(fitnesses)
        self.population = self.crossover(self.population)
        self.population = self.mutate(self.population)
        self.population = self.apply_drift(self.population)
        self.population = self.apply_genetic_drift(self.population)

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.population[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return parents

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                offspring[i] = (parent1 + parent2) / 2
        return offspring

    def mutate(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def apply_drift(self, population):
        drift = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        population = population + drift * (1 - self.drift_rate)
        return np.clip(population, -5.0, 5.0)

    def apply_genetic_drift(self, population):
        population = population + np.random.normal(0, 1, (self.population_size, self.dim)) * self.drift_rate
        return np.clip(population, -5.0, 5.0)

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

class DifferentialEvolution:
    def __init__(self, population_size, dim):
        self.population_size = population_size
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.diff = np.zeros((self.population_size, self.dim))

    def update(self, func):
        self.population = np.array([self.population[np.argsort(func(self.population))]])
        self.diff = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.population_size):
                if i!= j:
                    self.diff[i] += func(self.population[j]) - func(self.population[i])
        self.diff /= self.population_size
        self.population = self.population + self.diff * np.random.uniform(0.5, 1.5, (self.population_size, self.dim))

class GeneticDrift:
    def __init__(self, population_size, dim):
        self.population_size = population_size
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

# Example usage
def func(x):
    return np.sum(x**2)

ddea = DiversityDrivenEvolutionaryAlgorithm(budget=100, dim=10)
optimal_solution = ddea(func)
print(optimal_solution)