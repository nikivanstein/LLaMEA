import numpy as np
import random
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Evaluate the function for each member of the population
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        results = [func(x) for x in random.sample(bounds, self.budget)]
        
        # Select the best solution based on the budget
        best_idx = np.argmin(np.abs(results))
        best_func = results[best_idx]
        best_x = bounds[best_idx]
        
        # Update the population
        self.population.append((best_func, best_x))

        # Return the best function and its corresponding x-value
        return best_func, best_x

    def run(self):
        # Run the optimization algorithm for a specified number of generations
        for _ in range(100):
            # Select the best solution from the current population
            func, x = self.population[np.argmin(np.abs([x for _, x in self.population]))]
            
            # Refine the strategy by changing the probability of selecting the best solution
            if random.random() < 0.45:
                # Select the best solution based on the budget
                best_idx = np.argmin(np.abs([x for _, x in self.population]))
                best_func = results[best_idx]
                best_x = bounds[best_idx]
            else:
                # Select a random solution from the search space
                best_idx = random.randint(0, self.dim - 1)
                best_func = results[best_idx]
                best_x = bounds[best_idx]
            
            # Update the population
            self.population.append((best_func, best_x))

# Create an instance of the BlackBoxOptimizer class
optimizer = BlackBoxOptimizer(100, 10)

# Run the optimization algorithm
optimizer.run()

# Print the best function and its corresponding x-value
best_func, best_x = optimizer.population[np.argmin(np.abs([x for _, x in optimizer.population]))]
print("Best Function:", best_func)
print("Best X-Value:", best_x)