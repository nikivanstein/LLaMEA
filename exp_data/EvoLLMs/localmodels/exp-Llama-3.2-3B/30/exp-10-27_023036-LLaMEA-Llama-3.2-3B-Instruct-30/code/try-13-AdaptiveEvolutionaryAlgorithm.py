import numpy as np
import random

class AdaptiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)
        self.best_individual = self.population[0]
        self.best_fitness = self.fitness_values[0]

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the population
            self.fitness_values = func(self.population)
            # Select the fittest individuals
            fitness_order = np.argsort(self.fitness_values)
            self.population = self.population[fitness_order]
            self.population = self.population[:self.population_size//2]
            # Create a new generation by crossover and mutation
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1 = self.population[i]
                parent2 = self.population[(i+1)%self.population_size]
                child = parent1 + parent2
                child = child + np.random.uniform(-0.5, 0.5, size=self.dim)
                new_population[i] = child
            # Update the population
            self.population = new_population
            # Update the best individual
            self.best_individual = self.population[np.argmax(self.fitness_values)]
            self.best_fitness = np.max(self.fitness_values)

            # Adaptive probability update
            if np.random.rand() < 0.3:
                # Randomly replace a fraction of the population
                replace_indices = np.random.choice(self.population_size, size=int(self.population_size*0.3), replace=False)
                self.population[replace_indices] = self.population[np.random.choice(self.population_size, size=self.population_size, replace=True)]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
algorithm = AdaptiveEvolutionaryAlgorithm(budget, dim)
best_individual = algorithm(func)
print("Best individual:", best_individual)
print("Best fitness:", func(best_individual))