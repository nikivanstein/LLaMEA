# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.iteration_count = 0

    def __call__(self, func, exploration_strategy=None):
        def inner(x):
            return func(x)

        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)

        if exploration_strategy:
            x = exploration_strategy(x)

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
        
        # Update the iteration count
        self.iteration_count += 1

        # Refine the strategy based on the iteration count
        if self.iteration_count < 10:
            self.explore_strategy(x)
        elif self.iteration_count < 20:
            self.exploration_strategy(x)
        else:
            self.breeding_strategy(x)

        return x

    def explore_strategy(self, x):
        # Simple strategy: move towards the center of the search space
        x += np.random.uniform(-0.5, 0.5, self.dim)

    def exploration_strategy(self, x):
        # More complex strategy: use a combination of gradient descent and exploration
        learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
        dx = -np.dot(x - inner(x), np.gradient(y))
        x += learning_rate * dx
        exploration_rate = self.explore_rate / self.max_explore_count
        if np.random.rand() < exploration_rate:
            x += np.random.uniform(-0.5, 0.5, self.dim)

    def breeding_strategy(self, x):
        # Breed two individuals with different strategies
        child1 = self.evaluate_fitness(self.evaluate_fitness(x))
        child2 = self.evaluate_fitness(self.evaluate_fitness(np.random.uniform(-5.0, 5.0, self.dim)))
        return np.random.choice([child1, child2], p=[0.5, 0.5])

    def evaluate_fitness(self, func):
        # Evaluate a function at an individual
        return func(self.evaluate_individual(func))