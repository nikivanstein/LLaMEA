import numpy as np
from collections import deque

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.explore_history = deque(maxlen=self.max_explore_count)

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
            
            # Store the current exploration history
            self.explore_history.append((x, y))

        # Refine the strategy based on the exploration history
        new_individual = None
        if self.explore_history:
            # Get the last 10 exploration histories
            histories = self.explore_history[-10:]
            
            # Find the individual with the highest fitness
            best_individual, best_fitness = max(histories, key=lambda x: x[1])
            
            # Refine the strategy by changing the direction of the best individual
            new_individual = best_individual
            direction = np.array(best_individual) - np.array(new_individual)
            new_individual += direction * 0.1
            
            # Update the exploration history
            self.explore_history.append((new_individual, best_fitness))

        return x, new_individual

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Refines the strategy by changing the direction of the best individual based on exploration history