import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.budget_functions = 0
        self.population_size = 100
        self.population = None

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def select_individual(self, func, budget):
        # Select a random individual from the population
        individual = random.choices(range(self.population_size), k=1)[0]
        # Evaluate the function at the selected individual
        evaluation = func(individual)
        # Update the individual and its evaluation
        individual_evaluations = [evaluation]
        self.population[individual] = (individual, evaluation)
        # Update the best individual and its evaluation
        best_individual = None
        best_evaluation = float('-inf')
        for i, (ind, eval) in enumerate(self.population):
            if eval > best_evaluation:
                best_individual = ind
                best_evaluation = eval
        # Update the population with the best individual and its evaluation
        self.population = {ind: (ind, eval) for ind, eval in self.population.items() if eval > best_evaluation}
        # Update the budget functions with the best individual and its evaluation
        self.budget_functions = sum([eval for _, eval in self.population.values()])
        # Refine the strategy
        new_individual = random.choices(range(self.population_size), k=1)[0]
        # Evaluate the function at the new individual
        evaluation = func(new_individual)
        # Update the new individual and its evaluation
        new_individual_evaluations = [evaluation]
        self.population[new_individual] = (new_individual, evaluation)
        # Update the best individual and its evaluation
        best_individual_evaluations = [best_evaluation]
        for i, (ind, eval) in enumerate(self.population):
            if eval > best_individual_evaluations[i]:
                best_individual_evaluations[i] = eval
        # Update the best individual and its evaluation
        best_individual = min(best_individual_evaluations)
        # Update the population with the best individual and its evaluation
        self.population = {ind: (ind, eval) for ind, eval in self.population.items() if eval > best_individual_evaluations[i]}
        # Update the budget functions with the best individual and its evaluation
        self.budget_functions = sum([eval for _, eval in self.population.values()])
        # Return the new individual and its evaluation
        return new_individual, evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Code: