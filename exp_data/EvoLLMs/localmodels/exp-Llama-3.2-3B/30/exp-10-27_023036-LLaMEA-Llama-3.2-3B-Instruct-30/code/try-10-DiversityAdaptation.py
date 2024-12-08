import numpy as np
import random

class DiversityAdaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def evaluate(self, func):
        for individual in self.population:
            func(individual)
        return np.mean([func(individual) for individual in self.population])

    def mutate(self, individual, mutation_prob=0.3):
        if random.random() < mutation_prob:
            idx = random.randint(0, self.dim - 1)
            individual[idx] = np.random.uniform(-5.0, 5.0)
        return individual

    def select(self, func):
        scores = [func(individual) for individual in self.population]
        selected_indices = np.argsort(scores)[-self.population_size:]
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def optimize(self, func):
        for _ in range(self.budget):
            self.population = self.select(func)
            self.population = [self.mutate(individual) for individual in self.population]
            self.population = [self.mutate(individual, mutation_prob=0.3) if random.random() < 0.3 else individual for individual in self.population]

def diversity_adaptation(func, budget, dim):
    algorithm = DiversityAdaptation(budget, dim)
    algorithm.optimize(func)
    return algorithm.population

# Example usage:
# func = lambda x: x[0]**2 + x[1]**2
# budget = 100
# dim = 2
# population = diversity_adaptation(func, budget, dim)
# print(population)