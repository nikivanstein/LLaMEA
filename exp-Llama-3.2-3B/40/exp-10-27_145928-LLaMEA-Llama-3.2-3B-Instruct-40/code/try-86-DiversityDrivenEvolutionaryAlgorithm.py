import numpy as np
import random

class DiversityDrivenEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.drift_probability = 0.4
        self.population = self.initialize_population()
        self.differential_evolution = DifferentialEvolution(self.population_size, self.dim, self.crossover_probability, self.mutation_probability)
        self.genetic_drift = GeneticDrift(self.population_size, self.drift_probability)

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        self.population = self.differential_evolution.evaluate(func)
        self.population = self.genetic_drift.evaluate(self.population)
        self.population = self.select_parents(fitnesses)
        self.population = self.crossover(self.population)
        self.population = self.mutate(self.population)

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

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

class DifferentialEvolution:
    def __init__(self, population_size, dim, crossover_probability, mutation_probability):
        self.population_size = population_size
        self.dim = dim
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.diff_vector = np.zeros((self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        for i in range(self.population_size):
            self.diff_vector[i] = self.population[np.random.randint(0, self.population_size)] - self.population[i]
            if random.random() < self.crossover_probability:
                self.population[i] = (self.population[np.random.randint(0, self.population_size)] + self.population[np.random.randint(0, self.population_size)]) / 2
            if random.random() < self.mutation_probability:
                self.population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return fitnesses

class GeneticDrift:
    def __init__(self, population_size, drift_probability):
        self.population_size = population_size
        self.drift_probability = drift_probability
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, population):
        for i in range(self.population_size):
            if random.random() < self.drift_probability:
                population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return population

# Example usage
def func(x):
    return np.sum(x**2)

ddea = DiversityDrivenEvolutionaryAlgorithm(budget=100, dim=10)
optimal_solution = ddea(func)
print(optimal_solution)