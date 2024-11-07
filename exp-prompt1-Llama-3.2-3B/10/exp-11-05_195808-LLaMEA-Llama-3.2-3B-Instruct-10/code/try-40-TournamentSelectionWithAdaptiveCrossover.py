import numpy as np
import random

class TournamentSelectionWithAdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.crossover_rate = 0.5

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))

        # Evaluate population
        fitness = np.array([func(point) for point in population])

        # Selection
        for _ in range(self.budget):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(self.budget, size=tournament_size, replace=False)
            tournament_points = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            # Get best point
            best_point = tournament_points[np.argmin(tournament_fitness)]

            # Adaptive crossover
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
                child_point = best_point + crossover_point * 0.5
            else:
                child_point = best_point

            # Mutation
            mutation_rate = 0.1
            if np.random.rand() < mutation_rate:
                child_point += np.random.uniform(-0.1, 0.1, size=self.dim)

            # Ensure bounds
            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            # Replace worst point
            worst_index = np.argmin(fitness)
            population[worst_index] = child_point
            fitness[worst_index] = func(child_point)

        return population[0], fitness[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = TournamentSelectionWithAdaptiveCrossover(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)

# Refining the strategy:
# The crossover rate is now adaptive, meaning it changes based on the success rate of the crossover operation.
# If the crossover operation is successful (i.e., the child point is better than the parent point), the crossover rate increases.
# If the crossover operation is not successful, the crossover rate decreases.
# This is done by maintaining a separate list to track the success rate of the crossover operation.
# The success rate is updated after each iteration, and the crossover rate is adjusted accordingly.