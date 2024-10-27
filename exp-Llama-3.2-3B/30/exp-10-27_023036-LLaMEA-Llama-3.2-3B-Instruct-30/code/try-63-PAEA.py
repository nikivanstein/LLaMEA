import numpy as np
import random

class PAEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = []
        self.boundaries = [-5.0, 5.0]
        self.init_population()

    def init_population(self):
        for _ in range(self.population_size):
            individual = np.random.uniform(self.boundaries[0], self.boundaries[1], self.dim)
            self.population.append(individual)

    def __call__(self, func):
        if len(self.population) == 0:
            self.init_population()

        # Evaluate the fitness of each individual
        fitness = [func(individual) for individual in self.population]

        # Select the fittest individuals
        fittest_indices = np.argsort(fitness)[:self.population_size//2]
        self.population = [self.population[i] for i in fittest_indices]

        # Crossover and mutation
        new_population = []
        for _ in range(self.population_size//2):
            parent1, parent2 = random.sample(self.population, 2)
            child = parent1 + (parent2 - parent1) * np.random.uniform(0, 1, self.dim)
            new_population.append(child)

        # Adaptive mutation
        mutation_prob = 0.1
        for individual in new_population:
            if np.random.rand() < mutation_prob:
                individual += np.random.uniform(-0.1, 0.1, self.dim)
                individual = np.clip(individual, self.boundaries[0], self.boundaries[1])

        # Update the population
        self.population = new_population

        # Check if the budget is reached
        if len(self.population) >= self.budget:
            return max(fitness)

        # Replace the least fit individual with a new one
        least_fit_individual = min(self.population, key=lambda x: func(x))
        new_individual = np.random.uniform(self.boundaries[0], self.boundaries[1], self.dim)
        new_individual = np.clip(new_individual, self.boundaries[0], self.boundaries[1])
        self.population[self.population.index(least_fit_individual)] = new_individual

        return None

# Example usage:
def func(x):
    return np.sum(x**2)

paea = PAEA(100, 5)
print(paea(func))