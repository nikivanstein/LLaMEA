import numpy as np
import random

class MultiObjectiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.population = self.initialize_population()
        self.fitness_values = np.zeros((self.population_size, 1))

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitness_values = func(self.population)
        self.fitness_values = np.concatenate((self.fitness_values, fitness_values), axis=1)
        self.population = self.select_parents(fitness_values)
        self.population = self.crossover(self.population)
        self.population = self.mutate(self.population)

    def select_parents(self, fitness_values):
        fitness_values = np.array(fitness_values)
        parents = np.array([self.population[np.argsort(fitness_values[:, 0])[:int(self.population_size/2)]]])
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
        fitness_values = self.fitness_values[:, 1:]
        return np.argmin(fitness_values, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

mode = MultiObjectiveDifferentialEvolution(budget=100, dim=10)
optimal_solution = mode(func)
print(optimal_solution)