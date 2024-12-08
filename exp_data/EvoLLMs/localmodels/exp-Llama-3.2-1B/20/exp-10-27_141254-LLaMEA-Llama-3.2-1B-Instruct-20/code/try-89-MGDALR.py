import numpy as np
from scipy.optimize import minimize

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

    def select_strategy(self, individual):
        # Refine the strategy by changing the individual lines
        # to refine its search direction
        new_individual = individual.copy()
        
        if np.random.rand() < 0.2:
            # Change the direction of the individual
            new_individual[-1] += np.random.uniform(-0.5, 0.5)
        
        return new_individual

    def optimize(self, func):
        # Use the selected strategy to refine the individual
        # and optimize the function
        new_individual = self.select_strategy(func(self))
        
        # Optimize the function using the new individual
        result = minimize(func, new_individual, method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim)
        
        # Update the solution
        self.solution = new_individual
        self.score = result.fun
        
        return self.solution, self.score

# Initialize the algorithm
mgdalr = MGDALR(budget=100, dim=10)

# Define a black box function
def func(x):
    return x**2 + 2*x + 1

# Optimize the function
solution, score = mgdalr.optimize(func)

# Print the result
print(f"Solution: {solution}")
print(f"Score: {score}")