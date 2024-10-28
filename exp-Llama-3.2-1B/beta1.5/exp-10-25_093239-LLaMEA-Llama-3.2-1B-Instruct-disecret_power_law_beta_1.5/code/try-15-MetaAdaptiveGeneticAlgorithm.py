import random
import math
import numpy as np

class MetaAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size
        self.population_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')

    def __call__(self, func, num_generations=100):
        # Initialize the population history
        self.population_history = [(self.best_individual, self.best_fitness)]

        for generation in range(num_generations):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individual = max(self.population, key=lambda x: self.fitnesses[x])

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Update the best individual and its fitness
            if fitness > self.best_fitness:
                self.best_individual = next_individual
                self.best_fitness = fitness
                self.population_history.append((self.best_individual, self.best_fitness))

        # Return the best individual
        return self.best_individual

# One-line description: "Meta-Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.

# Example usage:
if __name__ == "__main__":
    # Define the function to optimize
    def func(x):
        return np.sin(x)

    # Initialize the Meta-Adaptive Genetic Algorithm
    meta_adaptive_algorithm = MetaAdaptiveGeneticAlgorithm(100, 10)

    # Optimize the function
    best_individual = meta_adaptive_algorithm(__call__(func))

    # Print the best individual
    print("Best Individual:", best_individual)
    print("Best Fitness:", meta_adaptive_algorithm.best_fitness)