import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation_cooling(self, func, population_size, mutation_rate, cooling_rate):
        # Initialize the population
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(population_size)]

        # Initialize the best individual
        best_individual = None
        best_fitness = -np.inf

        # Iterate over the population
        for _ in range(1000):
            # Evaluate the fitness of each individual
            fitnesses = [self(func(individual)) for individual in population]

            # Select the fittest individuals
            fittest_individuals = [individual for individual, fitness in zip(population, fitnesses) if fitness == max(fitnesses)]

            # Generate a new population
            new_population = []
            for _ in range(population_size):
                # Select two parents from the fittest individuals
                parent1, parent2 = random.sample(fittest_individuals, 2)

                # Generate a new individual by iterated permutation and cooling
                new_individual = self.iterated_permutation_cooling(func, population_size, mutation_rate, cooling_rate)
                new_population.append(new_individual)

            # Replace the old population with the new one
            population = new_population

            # Update the best individual
            if best_fitness < max(fitnesses):
                best_individual = fittest_individuals[0]
                best_fitness = max(fitnesses)

        # Return the best individual
        return best_individual

# One-line description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 