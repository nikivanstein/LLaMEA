import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def novel_bbo(self, func, budget, dim):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]
        # Initialize the best point and its fitness
        best_point = None
        best_fitness = float('-inf')
        # Iterate through the population
        for _ in range(budget):
            # Evaluate the fitness of each individual
            fitnesses = [self.func_evals() for individual in population]
            # Find the individual with the highest fitness
            best_individual = population[np.argmax(fitnesses)]
            # Update the best point and its fitness
            best_point = best_individual
            best_fitness = max(best_fitness, fitnesses[-1])
            # Refine the strategy
            if np.random.rand() < 0.45:
                # Increase the step size
                step_size = 0.1
            else:
                # Decrease the step size
                step_size = 0.05
            # Update the point using the new strategy
            new_point = best_point + np.random.uniform(-step_size, step_size, dim)
            # Evaluate the fitness of the new point
            new_fitness = self.func_evals() - 1  # subtract 1 to avoid overfitting
            # Add the new point to the population
            population.append(new_point)
            # Update the best point and its fitness
            best_point = new_point
            best_fitness = new_fitness
        # Return the best point found
        return best_point