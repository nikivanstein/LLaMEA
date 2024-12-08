import numpy as np
import random
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_probability = 0.1
        self.crossover_probability = 0.7
        self.mutator = random.Random()

    def __call__(self, func):
        # Initialize the population with random points in the search space
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Run the differential evolution algorithm
        for _ in range(self.budget):
            # Evaluate the fitness of each point in the population
            fitness = [func(point) for point in population]

            # Select the fittest points
            fittest_points = np.array(population[np.argsort(fitness)])

            # Perform crossover and mutation
            new_population = []
            for i in range(self.population_size):
                # Select two parents
                parent1 = fittest_points[i]
                parent2 = fittest_points[self.mutator.randint(0, self.population_size - 1)]

                # Perform crossover
                child = parent1 + (parent2 - parent1) * self.crossover_probability

                # Perform mutation
                if self.mutator.random() < self.mutation_probability:
                    child += np.random.uniform(-0.1, 0.1, self.dim)

                new_population.append(child)

            # Replace the old population with the new one
            population = np.array(new_population)

        # Return the best point in the final population
        return np.min(population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
he = HybridEvolutionaryAlgorithm(budget, dim)

best_point = he(func)
print("Best point:", best_point)