import numpy as np
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

    def __str__(self):
        return f"MGDALR: Black Box Optimization using Gradient Descent and Differential Evolution"

    def select_strategy(self):
        # Select a random individual from the current population
        current_individuals = [self.f(individual) for individual in self.population]
        selected_individual = np.random.choice(current_individuals, 1, replace=False)[0]
        
        # Refine the strategy by changing the individual's direction
        new_individual = selected_individual
        new_direction = self.gradient_descent(new_individual)
        
        # Update the individual's direction
        new_individual = self.f(new_individual)
        
        # Refine the strategy again by changing the individual's direction
        new_direction = self.gradient_descent(new_individual)
        new_individual = self.f(new_individual)
        
        return new_individual, new_direction

    def gradient_descent(self, individual):
        # Compute the gradient of the function at the individual
        gradient = np.gradient(self.f(individual))
        
        # Learn a new direction using gradient descent
        learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
        direction = -np.dot(individual - self.f(individual), gradient) / np.linalg.norm(gradient)
        
        return direction

    def f(self, individual):
        # Evaluate the function at the individual
        return self.budget * np.sum(individual)

# Description: Black Box Optimization using Gradient Descent and Differential Evolution
# Code: 