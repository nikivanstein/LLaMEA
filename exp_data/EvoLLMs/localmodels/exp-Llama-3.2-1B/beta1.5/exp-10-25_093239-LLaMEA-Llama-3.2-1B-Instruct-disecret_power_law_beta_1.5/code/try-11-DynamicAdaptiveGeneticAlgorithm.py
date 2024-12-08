import random
import math

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size
        self.population_history = []

    def __call__(self, func):
        for _ in range(self.budget):
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

            # Store the history of population
            self.population_history.append(self.population)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Also refine the strategy by changing the individual lines of the selected solution
        if self.population_size > 10:
            new_individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            for _ in range(self.population_size - 1):
                new_individual.append(random.uniform(-5.0, 5.0))
            new_individual = max(self.population, key=lambda x: self.fitnesses[x])
            return new_individual
        else:
            # Refine the strategy by changing the individual lines of the selected solution
            for _ in range(self.population_size):
                new_individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
                for _ in range(self.dim - 1):
                    new_individual.append(random.uniform(-5.0, 5.0))
                new_individual.append(random.uniform(-5.0, 5.0))
                new_individual = max(self.population, key=lambda x: self.fitnesses[x])
                return new_individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Fitness Refining"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting, and also refines its strategy