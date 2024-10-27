import numpy as np
import random

class DifferentialEvolutionWithAdaptiveLearningRate:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.best_individual = np.zeros(self.dim)
        self.best_fitness = -np.inf
        self.adaptive_learning_rate = 0.15

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness = func(self.population)

            # Select the fittest individuals
            parents = np.argsort(self.fitness)[:-int(self.population_size/2):-1]
            self.population = self.population[parents]

            # Perform differential evolution with adaptive learning rate
            for i in range(self.population_size):
                for j in range(1, self.population_size):
                    if random.random() < self.adaptive_learning_rate:
                        mutation = np.random.uniform(-0.1, 0.1, self.dim)
                    else:
                        mutation = np.zeros(self.dim)
                    self.population[i] += self.adaptive_learning_rate * (self.population[j] - self.population[i]) * mutation
                    self.population[i] = np.clip(self.population[i], -5.0, 5.0)

            # Calculate the fitness of the updated population
            self.fitness = func(self.population)

            # Update the population and the best individual
            self.population = self.population[np.argsort(self.fitness)]
            self.best_individual = self.population[0]
            self.best_fitness = func(self.best_individual)

            # Check for convergence
            if self.best_fitness > self.best_fitness:
                break

        return self.best_individual, self.best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = DifferentialEvolutionWithAdaptiveLearningRate(budget, dim)
best_individual, best_fitness = optimizer(func)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)