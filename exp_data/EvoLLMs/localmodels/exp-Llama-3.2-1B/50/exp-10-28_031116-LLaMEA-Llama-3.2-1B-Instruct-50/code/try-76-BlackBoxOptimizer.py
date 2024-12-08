import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Evaluate the black box function for each possible point in the search space
        points = np.linspace(-5.0, 5.0, self.dim)[:, None]
        evaluations = np.array([func(point) for point in points])

        # Select the best point based on the budget
        best_point = np.argmax(evaluations)
        best_func_value = evaluations[best_point]

        # Refine the strategy based on the probability of the selected point
        probability = 1.0 / self.budget
        if random.random() < probability:
            best_point = np.argmax(evaluations)
            best_func_value = evaluations[best_point]

        # Add the best point to the population
        self.population.append((best_point, best_func_value))

        # Calculate the fitness score based on the best point
        fitness_score = 1 / best_func_value
        self.fitness_scores.append(fitness_score)

    def select_solution(self):
        # Select the best solution based on the fitness score
        best_solution = self.population[np.argmax(self.fitness_scores)]
        best_func_value = best_solution[1]

        # Refine the strategy based on the probability of the selected solution
        probability = 1.0 / self.budget
        if random.random() < probability:
            best_func_value = self.population[np.argmax(self.fitness_scores)][1]

        return best_func_value

    def mutate(self, func, point, mutation_rate):
        # Mutate the black box function at the selected point
        new_func_value = func(point)
        if random.random() < mutation_rate:
            new_point = np.random.uniform(-5.0, 5.0, self.dim)[:, None]
            new_func_value = func(new_point)

        return new_func_value, new_point

# One-line description: Novel metaheuristic algorithm for solving black box optimization problems using a novel strategy that incorporates the probability of the selected point and a mutation rate.
# Code: 