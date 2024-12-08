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
        
        return x

class BBOOptimizer:
    def __init__(self, budget, dim, individual):
        self.budget = budget
        self.dim = dim
        self.individual = individual

    def __call__(self, func):
        # Refine the individual's strategy based on its fitness
        if self.individual.lines:
            # If the individual has multiple lines, refine each line separately
            for i in range(self.individual.lines):
                self.individual.lines[i].refine(func)
        else:
            # If the individual has only one line, refine it
            self.individual.refine(func)

    def generate_individual(self, func, dim):
        # Generate an initial individual with random values
        x = np.random.uniform(-5.0, 5.0, dim)
        return MGDALR(self.budget, dim)(func)(x)

# Generate a BBOOptimizer instance with 24 noiseless functions
optimizer = BBOOptimizer(24, 5, MGDALR(100, 5))

# Define a black box function
def func(x):
    return np.sin(x[0]) + np.cos(x[1])

# Evaluate the fitness of the initial individual
individual = optimizer.generate_individual(func, 5)
optimizer.__call__(func)(individual)
print("Fitness:", individual.fitness)