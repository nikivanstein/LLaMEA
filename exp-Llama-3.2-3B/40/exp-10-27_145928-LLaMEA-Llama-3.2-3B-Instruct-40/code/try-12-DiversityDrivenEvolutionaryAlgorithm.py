import numpy as np
import random

class DiversityDrivenEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
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
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.f = None
        self.x = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        if self.f is None:
            self.f = func
        fitnesses = self.f(self.x)
        self.x = self.select_parents(fitnesses)
        self.x = self.crossover(self.x)
        self.x = self.mutate(self.x)

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.x[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return parents

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                vector = parent1 - parent2
                vector = vector / np.linalg.norm(vector) * 0.5
                offspring[i] = parent1 + vector
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
        return np.min(self.x, axis=0)

class HybridAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.differential_evolution = DifferentialEvolution(budget, dim)
        self.ddea = DiversityDrivenEvolutionaryAlgorithm(budget, dim)
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1

    def evaluate(self, func):
        self.differential_evolution.evaluate(func)
        fitnesses = func(self.differential_evolution.x)
        self.ddea.evaluate(func)
        self.ddea.population = self.crossover(self.ddea.population)
        self.ddea.population = self.ddea.mutate(self.ddea.population)
        return fitnesses

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                vector = parent1 - parent2
                vector = vector / np.linalg.norm(vector) * 0.5
                offspring[i] = parent1 + vector
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
        return np.min(self.ddea.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

hybrid = HybridAlgorithm(budget=100, dim=10)
optimal_solution = hybrid(func)
print(optimal_solution)