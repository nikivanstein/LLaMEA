# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
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
        
        def learn_new_direction(x, y):
            # Calculate the gradient of the function at x
            gradient = np.gradient(y)
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - y, gradient)
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                return x
        
        # Evaluate the function at the current x
        y = inner(x)
        
        # If we've reached the maximum number of iterations, stop exploring
        if self.explore_count >= self.max_explore_count:
            return x
        
        # Refine the strategy
        new_individual = learn_new_direction(x, y)
        
        # Return the new individual
        return new_individual

def evaluate_fitness(individual, func, logger):
    # Evaluate the function at the individual
    updated_individual = func(individual)
    
    # Update the logger
    logger.update_fitness(updated_individual)
    
    return updated_individual

def main():
    # Define the black box function
    def func(x):
        return np.sum(x**2)
    
    # Define the logger
    logger = MGDALR(100, 10)
    
    # Evaluate the function 100 times
    for _ in range(100):
        x = np.array([-5.0] * 10)
        new_individual = MGDALR(100, 10).__call__(func)
        evaluate_fitness(new_individual, func, logger)
    
    # Print the results
    print("MGDALR:")
    print("Score:", logger.score)
    
    # Select the best solution
    best_individual = MGDALR(100, 10).__call__(func)
    print("Best Individual:", best_individual)

if __name__ == "__main__":
    main()