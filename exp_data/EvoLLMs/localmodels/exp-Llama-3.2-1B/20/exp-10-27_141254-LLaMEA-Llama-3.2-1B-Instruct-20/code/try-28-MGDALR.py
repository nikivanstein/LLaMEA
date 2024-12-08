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
        self.exploration_strategy = 'gradient_descent'
        self.fitness_history = []

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
            if self.exploration_strategy == 'gradient_descent':
                learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                dx = -np.dot(x - inner(x), np.gradient(y))
                x += learning_rate * dx
            elif self.exploration_strategy == 'random_search':
                exploration_strategy = random.choice(['gradient_descent', 'random_search'])
                if exploration_strategy == 'gradient_descent':
                    learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                    dx = -np.dot(x - inner(x), np.gradient(y))
                    x += learning_rate * dx
                else:
                    exploration_strategy = 'random_search'
                    exploration_strategy = random.choice(['gradient_descent', 'random_search'])
                    dx = -np.dot(x - inner(x), np.random.rand(self.dim))
                    x += dx * 0.1
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

# One-line description with the main idea
# Evolutionary algorithm with adaptive learning rate and exploration strategy
# to optimize black box functions using gradient descent or random search
# with a probability of 0.2 to change the individual lines of the selected solution
# to refine its strategy
# 
# 
# 
# 
# 