import numpy as np
import random

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        # Refine the solution using a novel heuristic
        # Generate a new set of directions
        directions = np.random.uniform(-5.0, 5.0, (self.dim,))
        
        # Evaluate the fitness of the new directions
        new_fitness = []
        for direction in directions:
            new_individual = inner(x + direction)
            new_fitness.append(self.evaluate_fitness(new_individual))
        
        # Select the best direction based on the probability of change
        best_direction_index = np.argmax(new_fitness)
        best_direction = directions[best_direction_index]
        
        # Refine the solution using the best direction
        x += best_direction
        
        return x

def evaluate_fitness(individual, logger):
    # Evaluate the fitness of the individual using the black box function
    func = lambda x: individual(x)
    return func(individual, logger)

def gradient(func, x):
    # Compute the gradient of the function at x
    return np.gradient(func(x))

# Initialize the metaheuristic algorithm
mgdalr = MGDALR(1000, 10)
mgdalr.explore_rate = 0.1
mgdalr.learning_rate = 0.01
mgdalr.max_explore_count = 1000

# Update the solution using the metaheuristic algorithm
mgdalr.__call__(lambda x: x)

# Print the solution
print(mgdalr.x)