# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
from scipy.optimize import differential_evolution

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

    def optimize(self, func, bounds, max_iter=1000, tol=1e-6):
        res = differential_evolution(func, bounds, args=(self.dim,))
        if res.success:
            return res.x
        else:
            raise Exception(f"Optimization failed: {res.message}")

# Test the algorithm
func = np.vectorize(lambda x: x**2)  # Example function
bounds = [(-10, 10)] * 10  # Example bounds
mgdalr = MGDALR(100, 10)  # Example optimization

# Initialize the population of algorithms
alg_pop = [
    MGDALR(100, 10),  # MGDALR
    MGDALR(100, 10),  # Another MGDALR
    MGDALR(100, 10)  # Another MGDALR
]

# Evaluate the fitness of each algorithm
alg_fitness = []
for alg in alg_pop:
    fitness = alg.optimize(func, bounds, max_iter=1000, tol=1e-6)
    alg_fitness.append((alg, fitness))

# Print the results
print("MGDALR:", mgdalr)
print("Optimal solution:", mgdalr.optimize(func, bounds, max_iter=1000, tol=1e-6))
print("Fitness:", mgdalr.optimize(func, bounds, max_iter=1000, tol=1e-6))
print("Other algorithms:", alg_fitness)