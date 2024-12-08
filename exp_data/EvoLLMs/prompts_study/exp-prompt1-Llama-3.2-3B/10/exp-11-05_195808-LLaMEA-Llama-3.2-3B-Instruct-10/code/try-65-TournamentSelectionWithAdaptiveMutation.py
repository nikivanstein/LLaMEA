import numpy as np
import random

class TournamentSelectionWithAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_rate = 0.1
        self.mutation_bound = 0.1

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

            # Crossover
            crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            child_point = best_point + crossover_point * 0.5

            # Adaptive mutation
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.uniform(-self.mutation_bound, self.mutation_bound, size=self.dim)
                child_point += mutation
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
optimizer = TournamentSelectionWithAdaptiveMutation(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)

# Refining the strategy
# Only changed 10% of the code
# The mutation rate and bound are now adaptive, meaning they change over time
# This is done by introducing a new variable `mutation_history` to store the number of mutations
# that have occurred in the current iteration, and adjusting the mutation rate and bound accordingly
class TournamentSelectionWithAdaptiveMutationRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_rate = 0.1
        self.mutation_bound = 0.1
        self.mutation_history = []

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

            # Crossover
            crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            child_point = best_point + crossover_point * 0.5

            # Adaptive mutation
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.uniform(-self.mutation_bound, self.mutation_bound, size=self.dim)
                child_point += mutation
                child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            # Update mutation history
            if np.random.rand() < 0.5:
                self.mutation_history.append(1)
            else:
                self.mutation_history = [0] + self.mutation_history[:-1]

            # Adjust mutation rate and bound based on mutation history
            if len(self.mutation_history) > 10:
                self.mutation_rate -= self.mutation_history[-1] / 10
                self.mutation_bound -= self.mutation_history[-1] / 10

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
optimizer = TournamentSelectionWithAdaptiveMutationRefined(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)