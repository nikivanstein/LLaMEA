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
        
        # Refine the strategy by changing the direction of the individual
        new_individual = self.evaluate_fitness(new_individual)
        directions = self.get_directions(x, new_individual)
        if directions:
            # Change the direction of the individual with probability 0.2
            if random.random() < 0.2:
                x = self.update_individual(x, directions)
        
        return x

    def get_directions(self, x, new_individual):
        # Get the gradients of the function at the current and new individuals
        gradients = np.gradient(new_individual, self.dim)
        
        # Get the directions of the individual
        directions = np.array([-gradients[i] / np.linalg.norm(gradients) for i in range(self.dim)])
        
        return directions

    def update_individual(self, x, directions):
        # Update the individual with the new direction
        new_individual = x + directions
        new_individual = np.clip(new_individual, -5.0, 5.0)
        
        return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func = lambda x: self.f(x)
        return func(individual)

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# The algorithm optimizes the black box function using gradient descent and refinement of the strategy
# using the direction of the individual with probability 0.2