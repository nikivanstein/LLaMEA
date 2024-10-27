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
        
        # Refine the strategy using a novel hybrid metaheuristic
        new_individual = self.refine_strategy(x)
        
        return x, new_individual

    def refine_strategy(self, x):
        # Define the novelty function
        def novelty(x):
            return np.mean(np.abs(x - np.random.uniform(-5.0, 5.0, self.dim)))
        
        # Refine the strategy using novelty function
        new_individual = x
        for _ in range(self.budget):
            # Evaluate the novelty function at the current x
            novelty_value = novelty(new_individual)
            
            # If the novelty value is high, refine the strategy
            if novelty_value > 0.2:
                # Learn a new direction using gradient descent
                learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                dx = -np.dot(new_individual - inner(new_individual), np.gradient(novelty_value))
                new_individual += learning_rate * dx
        
        return new_individual

class MGALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        return MGDALR(self.budget, self.dim)(func)

# Test the code
def test_func(x):
    return x[0]**2 + x[1]**2

mgdalr = MGALR(100, 10)
new_individual = mgdalr(10)
print(new_individual)