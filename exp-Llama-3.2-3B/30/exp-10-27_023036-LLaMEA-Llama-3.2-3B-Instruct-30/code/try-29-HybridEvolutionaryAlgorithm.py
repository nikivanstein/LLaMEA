import numpy as np
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of each individual
            self.fitness_values = func(self.population)

            # Select parents using tournament selection
            parents = []
            for _ in range(self.population_size // 2):
                tournament = random.sample(range(self.population_size), 3)
                winner = np.argmax([self.fitness_values[i] for i in tournament])
                parents.append(self.population[winner])

            # Create offspring using crossover and mutation
            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(parents, 2)
                child = np.random.uniform(-5.0, 5.0, size=self.dim)
                child[np.random.choice(self.dim, size=2, replace=False)] = (parent1 + parent2) / 2
                child[np.random.choice(self.dim, size=1, replace=False)] *= 0.9 + np.random.uniform(-0.1, 0.1)
                offspring.append(child)

            # Replace the least fit individuals with the new offspring
            self.population = np.array([self.population[i] if self.fitness_values[i] < np.min(self.fitness_values) else child for i, child in enumerate(offspring)])

# Example usage
def func(x):
    return np.sum(x ** 2)

hybrid_ea = HybridEvolutionaryAlgorithm(budget=100, dim=10)
hybrid_ea(func)