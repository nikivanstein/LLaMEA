import numpy as np

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

    def select_new_individual(self, func):
        # Select a new individual based on the probability of being selected
        # This is a novel heuristic that combines exploration and exploitation
        probability = self.explore_rate
        if np.random.rand() < probability:
            # Explore the search space
            return np.random.uniform(-5.0, 5.0, self.dim)
        else:
            # Exploit the current solution
            return x

    def __call__(self, func):
        new_individual = self.select_new_individual(func)
        return self.__call__(new_individual)

# Example usage:
func = lambda x: x**2
mgdalr = MGDALR(100, 10)
mgdalr(func)