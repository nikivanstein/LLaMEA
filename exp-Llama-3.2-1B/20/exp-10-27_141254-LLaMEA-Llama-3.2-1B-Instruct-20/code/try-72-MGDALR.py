import numpy as np

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.adaptation_rate = 0.2

    def __call__(self, func, logger):
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
            
            # Update the exploration strategy based on the fitness values
            if np.mean(y) > np.mean([y[i] for i in range(self.dim)]) * 0.8:
                self.adaptation_rate += 0.01
            else:
                self.adaptation_rate -= 0.01
            
            # If the adaptation rate exceeds a certain threshold, adapt the exploration strategy
            if self.adaptation_rate > 0.1:
                self.explore_count = 0
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

# Description: Multi-Objective Genetic Algorithm with Adaptive Learning Rate and Exploration Strategy
# Code: 