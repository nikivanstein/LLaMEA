import numpy as np
import random

class Paea:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.selection_rate = 0.5
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def fitness(self, func, individual):
        return func(individual)

    def select(self, population):
        scores = [self.fitness(func, individual) for individual in population]
        sorted_indices = np.argsort(scores)
        selected_indices = sorted_indices[:int(self.population_size * self.selection_rate)]
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            child = parent1 + (parent2 - parent1) / 2
        else:
            child = parent2 + (parent1 - parent2) / 2
        return child

    def mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                mutated_individual[i] += np.random.uniform(-1.0, 1.0)
        return mutated_individual

    def __call__(self, func):
        for _ in range(self.budget):
            population = self.select(self.population)
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
        return self.population[np.argmin([self.fitness(func, individual) for individual in self.population])]

# Example usage:
def func(x):
    return np.sum(x**2)

paea = Paea(budget=100, dim=10)
result = paea(func)
print(result)