import numpy as np
import random
from collections import deque

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.history = deque(maxlen=10)

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def adapt(self, new_individual):
        # Get the fitness of the new individual
        fitness = self.evaluate_fitness(new_individual)

        # If the fitness is better than the best fitness found so far, update the best individual
        if fitness > self.history[-1][1]:
            self.history.append((new_individual, fitness))
            return new_individual
        else:
            # If the fitness is not better, return the best individual found so far
            best_individual, best_fitness = self.history[-1]
            return best_individual

# One-line description: "Adaptive Black Box Optimizer: A novel metaheuristic algorithm that adapts its search strategy based on the optimization history to efficiently solve black box optimization problems"

# Initialize the optimizer with a budget of 1000 evaluations and a dimension of 5
optimizer = AdaptiveBlackBoxOptimizer(1000, 5)

# Evaluate the function for 1000 times
for _ in range(1000):
    func = lambda x: np.sin(x)
    optimizer(new_individual=optimizer.adapt(func(x)))