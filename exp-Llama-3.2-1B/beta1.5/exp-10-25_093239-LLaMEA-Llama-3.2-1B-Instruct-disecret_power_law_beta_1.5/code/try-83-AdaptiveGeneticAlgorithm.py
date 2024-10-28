import random
import math
import copy

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func):
        # Define the mutation probability
        mutation_prob = 0.1

        # Define the selection strategy
        def selection_strategy(individual, fitness):
            # Select the individual with the highest fitness
            # Use a simple strategy: select the individual with the highest fitness
            return max(individual, key=fitness)

        # Define the mutation strategy
        def mutation_strategy(individual, fitness):
            # Randomly select a dimension to mutate
            dim_to_mutate = random.randint(0, self.dim - 1)

            # Randomly select a point in the dimension to mutate
            point = random.uniform(-5.0, 5.0)

            # Mutate the point
            new_individual = copy.deepcopy(individual)
            new_individual[dim_to_mutate] = point + random.uniform(-mutation_prob, mutation_prob)

            # Ensure the fitness stays within the bounds
            new_individual[dim_to_mutate] = min(max(new_individual[dim_to_mutate], -5.0), 5.0)

            # Return the new individual
            return new_individual

        # Run the algorithm
        for _ in range(self.budget):
            # Select the next individual based on the fitness and the dimension
            new_individual = selection_strategy(self.population[0], self.fitnesses[0])

            # Evaluate the function at the new individual
            fitness = func(new_individual)

            # Update the fitness and the population
            self.fitnesses[0] += fitness
            self.population[0] = new_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[0] = min(max(self.fitnesses[0], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

# One-line description: "Adaptive Genetic Algorithm with Dynamic Selection"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.