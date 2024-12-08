import numpy as np
import random
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def mutate(self, individual):
        # Randomly select an individual from the current population
        new_individual = individual[np.random.choice(len(individual))]

        # Generate a new individual by flipping a random bit
        new_individual = new_individual.copy()
        flip_index = random.randint(0, len(new_individual) - 1)
        new_individual[flip_index] = 1 - new_individual[flip_index]

        # Evaluate the new individual using the same function
        value = self(func, new_individual)

        # Check if the new individual has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the new individual as the mutated solution
            return new_individual
        else:
            # If the new individual has been evaluated within the budget, return the original individual
            return individual

    def evolve_population(self, population, mutation_rate):
        # Randomly select individuals from the population to mutate
        mutated_population = random.choices(population, weights=[individual for individual in population if individual not in population[:mutation_rate]], k=mutation_rate)

        # Evolve the mutated population using the same function
        mutated_population = [self(func, individual) for individual in mutated_population]

        # Evaluate the mutated population
        mutated_population_evaluations = [self.func_evaluations + 1 for individual in mutated_population]

        # Update the population
        mutated_population = [individual for individual, evaluation in zip(mutated_population, mutated_population_evaluations) if evaluation < self.budget]

        # Return the updated population
        return mutated_population

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"