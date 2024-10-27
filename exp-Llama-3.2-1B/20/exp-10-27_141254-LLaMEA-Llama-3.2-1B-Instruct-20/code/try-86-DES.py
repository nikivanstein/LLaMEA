import numpy as np

class DES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

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
        
        # Refine the strategy based on the progress of the optimization process
        if self.explore_count < self.max_explore_count / 2:
            logger.update_individual(new_individual = x)
        elif self.explore_count < self.max_explore_count / 2 + 1:
            logger.update_individual(new_individual = x + np.random.normal(0, 1, self.dim))
        else:
            logger.update_individual(new_individual = x + 2 * np.random.normal(0, 1, self.dim))
        
        return x

# Example usage
def func(x):
    return np.sum(x**2)

logger = DES(budget=100, dim=10).evaluate_fitness(func)