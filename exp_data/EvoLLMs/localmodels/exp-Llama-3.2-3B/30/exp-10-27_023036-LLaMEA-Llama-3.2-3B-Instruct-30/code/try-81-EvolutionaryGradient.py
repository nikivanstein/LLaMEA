import numpy as np
import random

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)
        self.best_individual = np.random.uniform(-5.0, 5.0, (1, self.dim))
        self.best_fitness = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Calculate fitness values
            self.fitness_values = func(self.population)

            # Select best individual
            self.best_individual = self.population[np.argmin(self.fitness_values)]
            self.best_fitness = np.min(self.fitness_values)

            # Create a new population by perturbing the best individual
            new_population = self.population.copy()
            for i in range(self.population_size):
                # Perturb the individual with a probability of 0.3
                if random.random() < 0.3:
                    new_population[i] += np.random.uniform(-0.1, 0.1, self.dim)

            # Replace the old population with the new one
            self.population = new_population

            # Update the best individual
            self.best_individual = self.population[np.argmin(self.fitness_values)]
            self.best_fitness = np.min(self.fitness_values)

            # Print the best individual and its fitness value
            print(f"Best individual: {self.best_individual}, Best fitness: {self.best_fitness}")

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = EvolutionaryGradient(budget, dim)
optimizer("func")