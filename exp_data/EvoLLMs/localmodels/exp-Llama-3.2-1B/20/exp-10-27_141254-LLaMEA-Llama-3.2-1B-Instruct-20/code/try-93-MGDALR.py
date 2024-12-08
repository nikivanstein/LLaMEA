# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
import numpy as np
import random
from collections import deque

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.population_size = 100
        self.population = deque(maxlen=self.population_size)

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
            
            # Refine the strategy by changing the direction
            if random.random() < 0.2:
                new_direction = np.random.uniform(-5.0, 5.0, self.dim)
                new_individual = x + self.explore_rate * new_direction
                self.population.append(new_individual)

        # Select the best individual based on the fitness
        fitness = [self.f(individual, self.logger) for individual in self.population]
        best_individual = self.population[np.argmax(fitness)]
        return best_individual

    def f(self, individual, logger):
        func = lambda x: individual(x)
        return func(individual)

# Initialize the algorithm with the BBOB test suite
alg = MGDALR(1000, 10)

# Add the current solution to the population
alg.population.append(alg.__call__(alg.func))

# Print the initial population
print("Initial population:")
print(alg.population)

# Run the optimization algorithm
for _ in range(100):
    alg.func()
    print("\nAfter iteration", _ + 1, ":", alg.population)