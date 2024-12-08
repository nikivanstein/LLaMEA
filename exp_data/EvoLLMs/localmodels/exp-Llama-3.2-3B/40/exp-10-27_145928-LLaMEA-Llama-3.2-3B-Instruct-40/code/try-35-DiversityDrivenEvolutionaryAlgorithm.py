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
                offspring[i] = (parent1 + parent2) / 2 + np.random.uniform(-0.2, 0.2, self.dim)
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

class DifferentialEvolutionAlgorithm:
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
                target = np.mean(parent1, axis=0)
                for j in range(self.dim):
                    if random.random() < 0.5:
                        offspring[i, j] = parent1[j] + (parent2[j] - parent1[j]) * np.random.uniform(-0.5, 0.5)
                    else:
                        offspring[i, j] = parent2[j] + (parent1[j] - parent2[j]) * np.random.uniform(-0.5, 0.5)
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

class HybridAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.differential_evolution = DifferentialEvolutionAlgorithm(budget, dim)
        self.diversity_driven_evolution = DiversityDrivenEvolutionaryAlgorithm(budget, dim)
        self.population = self.differential_evolution.initialize_population()

    def evaluate(self, func):
        self.differential_evolution.evaluate(func)
        fitnesses = func(self.differential_evolution.population)
        self.population = self.diversity_driven_evolution.select_parents(fitnesses)
        self.population = self.diversity_driven_evolution.crossover(self.population)
        self.population = self.diversity_driven_evolution.mutate(self.population)

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.population[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return parents

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < 0.8:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                target = np.mean(parent1, axis=0)
                for j in range(self.dim):
                    if random.random() < 0.5:
                        offspring[i, j] = parent1[j] + (parent2[j] - parent1[j]) * np.random.uniform(-0.2, 0.2)
                    else:
                        offspring[i, j] = parent2[j] + (parent1[j] - parent2[j]) * np.random.uniform(-0.2, 0.2)
            else:
                offspring[i] = random.choice(population)
        return offspring

    def mutate(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < 0.1:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

hybrid = HybridAlgorithm(budget=100, dim=10)
optimal_solution = hybrid(func)
print(optimal_solution)