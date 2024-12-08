# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize
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
        
        def evaluate_fitness(individual):
            return minimize(inner, individual, method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim, constraints={"type": "eq", "fun": lambda x: x[-1] - 5.0})
        
        # Initialize population with random individuals
        population = [evaluate_fitness(random.uniform(-5.0, 5.0)) for _ in range(100)]
        
        while len(population) < self.budget:
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
            dx = -np.dot(x - evaluate_fitness(random.uniform(-5.0, 5.0)), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

# One-line description with main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# This algorithm combines the benefits of gradient-based optimization and genetic algorithms to efficiently explore the search space of a black box function.
# 
# Code:
# ```python
# ```python