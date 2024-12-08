import numpy as np

class GWE:
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
        
        # Evaluate the function at the current x
        y = inner(x)
        
        # If we've reached the maximum number of iterations, stop exploring
        if self.explore_count >= self.max_explore_count:
            return x
        
        # If we've reached the upper bound, stop exploring
        if x[-1] >= 5.0:
            return x
        
        # Learn a new direction using gradient descent
        learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
        dx = -np.dot(x - inner(x), np.gradient(y))
        x += learning_rate * dx
        
        # Update the exploration count
        self.explore_count += 1
        
        return x

    def update_individual(self, individual):
        # Refine the strategy by changing the individual lines of the selected solution
        # to refine its strategy
        new_individual = individual
        if np.random.rand() < 0.2:
            # Increase the learning rate for more aggressive exploration
            self.learning_rate *= 1.1
            new_individual = np.array([np.random.uniform(-5.0, 5.0) for _ in range(self.dim)])
        elif np.random.rand() < 0.2:
            # Decrease the learning rate for more conservative exploration
            self.learning_rate *= 0.9
            new_individual = np.array([np.random.uniform(-5.0, 5.0) for _ in range(self.dim)])
        else:
            # Maintain the current learning rate for balanced exploration
            new_individual = np.array([np.random.uniform(-5.0, 5.0) for _ in range(self.dim)])
        
        return new_individual