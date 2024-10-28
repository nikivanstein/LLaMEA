import numpy as np
import random

class DiversityDrivenEvolutionaryAlgorithmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.differential_evolution_probability = 0.2
        self.gaussian_noise_probability = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        self.population = self.select_parents(fitnesses)
        self.population = self.crossover(self.population)
        self.population = self.mutate(self.population)
        self.population = self.differential_evolution(self.population)
        self.population = self.add_gaussian_noise(self.population)

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

    def differential_evolution(self, population):
        for i in range(self.population_size):
            if random.random() < self.differential_evolution_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child = (parent1 + parent2) / 2
                child = child + np.random.uniform(-1.0, 1.0, self.dim)
                population[i] = child
        return population

    def add_gaussian_noise(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < self.gaussian_noise_probability:
                mutated_population[i] += np.random.normal(0, 0.1, self.dim)
        return mutated_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

ddea_de = DiversityDrivenEvolutionaryAlgorithmDE(budget=100, dim=10)
optimal_solution = ddea_de(func)
print(optimal_solution)