import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.2
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.learning_rate_adaptation = 0.5

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
            if learning_rate < self.learning_rate_adaptation:
                dx = -np.dot(x - inner(x), np.gradient(y))
            else:
                dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

# Example usage:
func = lambda x: x**2
novel_metaheuristic = NovelMetaheuristic(100, 10)
solution = novel_metaheuristic(func, logger)