import numpy as np
from scipy.optimize import minimize
import random
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        # Initialize population with random initial guesses
        population = [copy.deepcopy(initial_guess) for _ in range(100)]

        for _ in range(iterations):
            for i, population_i in enumerate(population):
                # Evaluate the fitness of each individual
                fitness = self.func(population_i)

                # Select the best individual based on the probability of 0.45
                selected_individual = population_i[np.random.choice(len(population_i), p=[0.45]*len(population_i))]

                # Create a new individual by refining the selected individual
                new_individual = selected_individual.copy()
                for _ in range(10):  # Number of iterations for refinement
                    # Evaluate the fitness of the new individual
                    fitness = self.func(new_individual)

                    # Refine the new individual based on the probability of 0.45
                    new_individual = copy.deepcopy(new_individual)
                    new_individual[np.random.choice(new_individual.shape[0], p=[0.45]*new_individual.shape[0])] += random.uniform(-0.01, 0.01)

                    # Check if the new individual is within the search space
                    if new_individual[0] < -5.0 or new_individual[0] > 5.0 or new_individual[1] < -5.0 or new_individual[1] > 5.0:
                        break

                # Replace the selected individual with the new individual
                population[i] = new_individual

        # Return the best individual
        return population[np.argmax([self.func(individual) for individual in population])], self.func(population[np.argmax([self.func(individual) for individual in population])))

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 