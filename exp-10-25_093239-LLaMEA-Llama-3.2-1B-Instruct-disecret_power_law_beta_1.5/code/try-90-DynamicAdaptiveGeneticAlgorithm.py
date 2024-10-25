import random
import math

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size
        self.refinement_factor = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
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

            # Refine the strategy by changing the selection probability
            if random.random() < self.refinement_factor:
                # Use a more aggressive strategy to refine the solution
                new_individual = self.select_next_individual_with_aggressive_strategy()

            # Update the population with the new individual
            self.population[self.population_size - 1] = new_individual

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        return max(self.population, key=lambda x: self.fitnesses[x])

    def select_next_individual_with_aggressive_strategy(self):
        # Select the next individual based on the fitness and the dimension,
        # but with a more aggressive strategy to refine the solution
        # Use a simple strategy: select the individual with the highest fitness
        return max(self.population, key=lambda x: self.fitnesses[x])

# One-line description: "Evolutionary Algorithm with Adaptive Sampling and Fitness Refinement"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting, and also refines its strategy using a more aggressive approach.