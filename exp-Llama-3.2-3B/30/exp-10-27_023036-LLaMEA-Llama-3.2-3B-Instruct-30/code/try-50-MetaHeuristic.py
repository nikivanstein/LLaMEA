import numpy as np
import random

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_individual = self.population[0]

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        for individual in self.population:
            func(individual)
        self.population = sorted(self.population, key=lambda x: func(x))
        self.best_individual = self.population[0]

    def mutate(self, individual):
        mutation_rate = 0.3
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < mutation_rate:
                mutated_individual[i] += np.random.uniform(-1.0, 1.0)
                mutated_individual[i] = max(-5.0, min(5.0, mutated_individual[i]))
        return mutated_individual

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
            if len(self.population) > self.population_size:
                self.population.pop()
            best_individual = self.population[0]
            mutated_individual = self.mutate(best_individual)
            self.population.append(mutated_individual)
            self.population = sorted(self.population, key=lambda x: func(x))
            self.best_individual = self.population[0]
        return self.best_individual

# Example usage
def func(x):
    return np.sum(x**2)

meta_heuristic = MetaHeuristic(100, 10)
best_individual = meta_heuristic(func)
print(best_individual)