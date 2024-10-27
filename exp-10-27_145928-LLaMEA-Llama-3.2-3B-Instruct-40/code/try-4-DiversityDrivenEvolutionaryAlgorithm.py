import numpy as np
import random

class DiversityDrivenEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.adaptive_diff_evolution_probability = 0.4
        self.population = self.initialize_population()
        self.diff_evolution = np.zeros((self.population_size, self.dim))

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        self.population = self.select_parents(fitnesses)
        self.population = self.crossover(self.population)
        self.population = self.mutate(self.population)
        if random.random() < self.adaptive_diff_evolution_probability:
            self.diff_evolution = self.adaptive_diff_evolution(self.population, self.diff_evolution)
            self.population = self.population + self.diff_evolution
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

    def adaptive_diff_evolution(self, population, diff_evolution):
        fitnesses = np.array([np.sum(ind**2) for ind in population])
        min_fitness = np.min(fitnesses)
        max_fitness = np.max(fitnesses)
        diff_evolution = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if fitnesses[i] < min_fitness + 0.2*(max_fitness - min_fitness):
                diff_evolution[i] = np.random.uniform(-1.0, 1.0, self.dim)
        return diff_evolution

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

ddea = DiversityDrivenEvolutionaryAlgorithm(budget=100, dim=10)
optimal_solution = ddea(func)
print(optimal_solution)