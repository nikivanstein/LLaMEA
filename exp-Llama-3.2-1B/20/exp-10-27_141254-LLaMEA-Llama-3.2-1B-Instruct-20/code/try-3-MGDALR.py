# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import differential_evolution

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

def select_strategy(individual, budget, dim):
    # Select a new strategy based on the individual's fitness
    if individual[0] > 0.5:
        # Randomly select a new strategy
        new_strategy = random.uniform(-1, 1)
    else:
        # Use a probability of 0.2 to change the individual's strategy
        new_strategy = individual[0] + 0.2 * random.uniform(-1, 1)
    
    # Refine the strategy using the selected individual
    refined_strategy = individual + [new_strategy]
    
    return refined_strategy

def optimize_func(func, refined_strategy):
    # Optimize the function using the refined strategy
    result = differential_evolution(func, [(x, y) for x, y in refined_strategy])
    
    return result.x

# Evaluate the fitness of the initial solution
initial_individual = np.array([-5.0] * 10)
initial_fitness = optimize_func(lambda x: x[0], initial_individual)

# Print the initial solution and its fitness
print(f"Initial Solution: {initial_individual} with fitness {initial_fitness}")

# Select a new strategy and refine the individual
selected_strategy = select_strategy(initial_individual, 100, 10)
refined_individual = select_strategy(selected_strategy, 100, 10)

# Optimize the function using the refined individual
optimized_individual = optimize_func(lambda x: x[0], refined_individual)

# Print the optimized solution and its fitness
print(f"Optimized Solution: {optimized_individual} with fitness {optimized_individual[0]}")