import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.iterations = 0
        self.temperature = 1.0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        # Randomly change one bit in the individual
        bit_index = random.randint(0, self.dim)
        individual[bit_index] = 1 - individual[bit_index]
        # If the bit is changed, recalculate the fitness
        if random.random() < 0.15:
            self.iterations += 1
            self.temperature *= 0.99
        return individual

    def simulate_annealing(self, initial_temperature):
        # Initialize the current temperature
        current_temperature = initial_temperature
        # Initialize the best point found so far
        best_point = self.search_space[0], self.search_space[1]
        # Iterate until the budget is reached
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
            # Calculate the new temperature
            new_temperature = current_temperature * 0.99
            # If the new temperature is less than 0.1, stop the simulation
            if new_temperature < 0.1:
                break
            # If the new temperature is greater than the current temperature, change the bit
            if random.random() < 0.15:
                bit_index = random.randint(0, self.dim)
                individual = self.mutate(point)
                # Recalculate the fitness
                if random.random() < 0.15:
                    self.iterations += 1
                    self.temperature *= 0.99
            # Update the best point found so far
            if func_value < best_point[0] or (func_value == best_point[0] and random.random() < 0.15):
                best_point = point
        # Return the best point found so far
        return best_point

# Example usage:
# Create an instance of the BlackBoxOptimizer
optimizer = BlackBoxOptimizer(100, 10)
# Optimize the function f(x) = x^2
best_point = optimizer.optimize(lambda x: x**2)
print(best_point)