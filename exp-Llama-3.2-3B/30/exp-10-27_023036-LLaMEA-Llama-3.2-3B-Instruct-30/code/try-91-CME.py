import numpy as np
import random
from scipy.optimize import differential_evolution

class CME:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_probability = 0.3
        self.mutation_probability = 0.1
        self.exploration_probability = 0.2

    def __call__(self, func):
        # Initialize the population
        population = np.array([np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)])

        # Run the optimization
        for _ in range(self.budget):
            # Evaluate the population
            scores = np.array([func(x) for x in population])

            # Select the best individuals
            best_individuals = population[np.argsort(scores)]

            # Crossover and mutation
            new_population = []
            for i in range(self.population_size):
                # Select two parents
                parent1 = best_individuals[i]
                parent2 = best_individuals[(i + 1) % self.population_size]

                # Crossover
                if random.random() < self.crossover_probability:
                    child = (parent1 + parent2) / 2
                else:
                    child = parent1

                # Mutation
                if random.random() < self.mutation_probability:
                    child += np.random.uniform(-1.0, 1.0, self.dim)

                # Add the child to the new population
                new_population.append(child)

            # Replace the old population with the new one
            population = np.array(new_population)

            # Explore the search space
            if random.random() < self.exploration_probability:
                # Randomly select a point in the search space
                exploration_point = np.random.uniform(-5.0, 5.0, self.dim)
                # Evaluate the exploration point
                scores = np.append(scores, func(exploration_point))
                # Add the exploration point to the population
                population = np.append(population, exploration_point)

        # Return the best individual
        scores = np.array([func(x) for x in population])
        best_individual = population[np.argsort(scores)]
        return best_individual[0]

# Example usage
def func(x):
    return np.sum(x**2)

cme = CME(budget=100, dim=10)
best_individual = cme(func)
print("Best individual:", best_individual)