import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveProbabilisticEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutate_prob = 0.1
        self.crossover_prob = 0.7
        self.population = self.initialize_population()

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        return population

    def fitness(self, func, x):
        return func(x)

    def evaluate(self, func):
        for _ in range(self.budget):
            fitnesses = [self.fitness(func, x) for x in self.population]
            best_idx = np.argmin(fitnesses)
            best_x = self.population[best_idx]
            best_f = fitnesses[best_idx]

            # Select parents
            parents = np.array([self.population[np.random.choice(self.population_size, 2, replace=False)] for _ in range(self.population_size)])

            # Crossover
            offspring = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1, parent2 = parents[i, :]
                if np.random.rand() < self.crossover_prob:
                    offspring[i] = (parent1 + parent2) / 2

            # Mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutate_prob:
                    offspring[i] += np.random.uniform(-1.0, 1.0, self.dim)

            # Replace worst individual
            worst_idx = np.argmin([self.fitness(func, x) for x in offspring])
            self.population[worst_idx] = offspring[worst_idx]

    def optimize(self, func):
        self.evaluate(func)
        return np.min([self.fitness(func, x) for x in self.population])

# Usage
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = AdaptiveProbabilisticEvolutionaryAlgorithm(budget, dim)
best_x = optimizer.optimize(func)
print("Best x:", best_x)
print("Best f:", func(best_x))