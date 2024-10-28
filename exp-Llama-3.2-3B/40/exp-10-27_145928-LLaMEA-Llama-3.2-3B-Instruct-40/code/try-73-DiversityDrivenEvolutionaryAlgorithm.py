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
                x1, x2 = parent1, parent2
                r = np.random.uniform(0, 1, self.dim)
                x1 = x1 + r * (parent2 - parent1)
                x2 = x2 - r * (parent2 - parent1)
                offspring[i] = (x1 + x2) / 2
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

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.differential_evolution = DifferentialEvolution(budget=self.budget, dim=self.dim)
        self.differential_evolution.population = self.differential_evolution.initialize_population()
        self.diversity_driven_evolutionary_algorithm = DiversityDrivenEvolutionaryAlgorithm(budget=self.budget, dim=self.dim)
        self.diversity_driven_evolutionary_algorithm.population = self.diversity_driven_evolutionary_algorithm.initialize_population()

    def evaluate(self, func):
        self.differential_evolution.evaluate(func)
        self.diversity_driven_evolutionary_algorithm.evaluate(func)

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.differential_evolution.population[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        parents = np.array([self.diversity_driven_evolutionary_algorithm.population[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return parents

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.differential_evolution.crossover_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                x1, x2 = parent1, parent2
                r = np.random.uniform(0, 1, self.dim)
                x1 = x1 + r * (parent2 - parent1)
                x2 = x2 - r * (parent2 - parent1)
                offspring[i] = (x1 + x2) / 2
        return offspring

    def mutate(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < self.differential_evolution.mutation_probability:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

hybrid_algorithm = HybridEvolutionaryAlgorithm(budget=100, dim=10)
optimal_solution = hybrid_algorithm(func)
print(optimal_solution)