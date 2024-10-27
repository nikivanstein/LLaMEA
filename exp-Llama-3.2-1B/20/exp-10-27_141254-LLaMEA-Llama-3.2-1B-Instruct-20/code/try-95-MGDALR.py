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

    def __call__(self, func, initial_individual):
        def inner(individual):
            return func(individual)
        
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
            
            # Learn a new direction using stochastic gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

class StochasticGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func, initial_individual):
        def inner(individual):
            return func(individual)
        
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
            
            # Learn a new direction using stochastic gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.random.rand(self.dim))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

def bbof(x, func, initial_individual, budget, dim):
    if isinstance(x, str):
        x = np.array([float(i) for i in x.split(',')])
    return MGDALR(budget, dim)(func, initial_individual)

def bbof_stochastic_gradient_descent(x, func, initial_individual, budget, dim):
    if isinstance(x, str):
        x = np.array([float(i) for i in x.split(',')])
    return StochasticGradientDescent(budget, dim)(func, initial_individual)

# Example usage:
def func(x):
    return np.sum(x**2)

initial_individual = np.array([-5.0] * 10)
x = bbof(func, initial_individual, initial_individual, 1000, 10)
print(x)

x = bbof_stochastic_gradient_descent(func, initial_individual, initial_individual, 1000, 10)
print(x)