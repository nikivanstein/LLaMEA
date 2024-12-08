# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, budget=100, step_size=0.1, cooling_rate=0.99):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < budget:
            # Initialize the current point and evaluation
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            evaluation = func(point)
            # Initialize the population
            population = [point, evaluation]
            # Initialize the best point and evaluation
            best_point = point
            best_evaluation = evaluation
            # Initialize the counter
            count = 0
            # Initialize the current point
            current_point = point
            # Initialize the current evaluation
            current_evaluation = evaluation
            # Initialize the population size
            population_size = 100
            # Initialize the best individual
            best_individual = point
            best_individual_evaluation = evaluation
            while count < budget:
                # Generate a new point using linear interpolation
                new_point = current_point + (current_point - point) * step_size
                # Evaluate the function at the new point
                new_evaluation = func(new_point)
                # Increment the population size
                population_size += 1
                # Check if the budget is reached
                if count + population_size > budget:
                    # If the budget is reached, return a default point and evaluation
                    return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))
                # If the new evaluation is better, update the best individual and evaluation
                if new_evaluation > current_evaluation:
                    best_point = new_point
                    best_evaluation = new_evaluation
                    best_individual = point
                    best_individual_evaluation = evaluation
                # If the new evaluation is not better, update the current point and evaluation
                else:
                    current_point = new_point
                    current_evaluation = new_evaluation
                    # If the current evaluation is better, update the best individual and evaluation
                    if current_evaluation > best_evaluation:
                        best_point = new_point
                        best_evaluation = current_evaluation
                        best_individual = point
                        best_individual_evaluation = evaluation
                # Increment the counter
                count += population_size
                # Update the current point
                current_point = new_point
            # Return the best individual and evaluation
            return best_individual, best_evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Current population of algorithms
black_box_optimizers = [
    BlackBoxOptimizer(100, 5),
    BlackBoxOptimizer(100, 10),
    BlackBoxOptimizer(100, 15),
    BlackBoxOptimizer(100, 20),
    BlackBoxOptimizer(100, 25),
    BlackBoxOptimizer(100, 30),
    BlackBoxOptimizer(100, 35),
    BlackBoxOptimizer(100, 40),
    BlackBoxOptimizer(100, 45),
    BlackBoxOptimizer(100, 50),
]

# Update the current population
black_box_optimizers[0] = BlackBoxOptimizer(100, 5)