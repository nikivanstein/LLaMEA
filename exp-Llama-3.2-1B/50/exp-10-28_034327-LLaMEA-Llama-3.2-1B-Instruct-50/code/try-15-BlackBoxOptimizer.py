import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0
        self.population_size = 100

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

    def iterated_permutation(self, func, budget, dim, iterations):
        # Initialize the population with random points
        population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(self.population_size)]
        
        # Iterate for the specified number of iterations
        for _ in range(iterations):
            # Initialize the best point and its value
            best_point = population[0]
            best_value = self.evaluate_fitness(best_point)
            
            # Iterate over the population
            for individual in population[1:]:
                # Evaluate the function at the individual
                value = self.evaluate_fitness(individual)
                
                # Check if the individual is better than the best point
                if value > best_value:
                    # Update the best point and its value
                    best_point = individual
                    best_value = value
                    
            # Update the population with the best point
            population = [best_point] + population[1:]
        
        # Return the best point found after the specified number of iterations
        return self.evaluate_fitness(population[0])

    def cooling_schedule(self, initial_value, final_value, cooling_rate, iterations):
        # Initialize the cooling rate and the final value
        cooling_rate = initial_value / final_value
        final_value = initial_value
        
        # Initialize the current value
        current_value = initial_value
        
        # Iterate for the specified number of iterations
        for _ in range(iterations):
            # Update the current value using the cooling schedule
            current_value *= cooling_rate
            
            # Check if the current value is within the bounds
            if -5.0 <= current_value <= 5.0:
                # If the current value is within bounds, return it
                return current_value
            else:
                # If the current value is not within bounds, update it
                current_value = np.random.uniform(-5.0, 5.0)
        
        # If the current value exceeds the final value, return the final value
        return final_value