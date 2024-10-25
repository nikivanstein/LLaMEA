import random
import math
import numpy as np

class AdaptiveBlackBoxOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func, iterations):
        for _ in range(iterations):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            new_individual = self.select_next_individual()

            # Evaluate the function at the new individual
            fitness = func(new_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = new_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and refine it using a probabilistic strategy
        # Refine the individual based on the fitness and the dimension
        # Use a simple probabilistic strategy: select the individual with the highest fitness
        # and refine it based on the probability of the individual being good
        probabilities = [self.fitnesses / self.fitnesses[self.population_size - 1] for _ in range(self.population_size)]
        best_index = np.argmax(probabilities)
        best_individual = self.population[best_index]
        new_individual = best_individual
        if random.random() < 0.05:
            # Refine the individual based on the fitness and the dimension
            # Use a simple probabilistic strategy: select the individual with the highest fitness
            # and refine it based on the probability of the individual being good
            probabilities = [self.fitnesses / self.fitnesses[self.population_size - 1] for _ in range(self.population_size)]
            best_index = np.argmax(probabilities)
            new_individual = self.population[best_index]
        return new_individual

# One-line description: "Adaptive Black Box Optimization using Dynamic Selection"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.