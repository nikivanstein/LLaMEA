import random
import numpy as np
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, strategy, iterations=100):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Initialize the current point and evaluation
            current_point = np.random.uniform(self.search_space[0], self.search_space[1])
            evaluation = func(current_point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and evaluation
            return current_point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def novel_metaheuristic(self, func, strategy, iterations=100):
        # Initialize the population with random points
        population = [copy.deepcopy(func) for _ in range(100)]
        # Initialize the best individual and its fitness
        best_individual = population[0]
        best_fitness = func(best_individual)

        # Run the metaheuristic algorithm
        for _ in range(iterations):
            # Select a strategy from the available strategies
            strategy_index = random.randint(0, len(strategy) - 1)
            # Apply the selected strategy to each individual
            for individual in population:
                # Evaluate the fitness of the individual using the selected strategy
                fitness = strategy[strategy_index](individual)
                # Update the individual's fitness and its point
                individual[0] = func(individual[0], fitness)
                individual[1] = fitness
            # Get the individual with the highest fitness
            best_individual = max(population, key=lambda individual: individual[1])

        # Return the best individual and its fitness
        return best_individual, best_fitness

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: