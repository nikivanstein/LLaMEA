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

    def update_individual(self, individual):
        # Refine the individual's strategy
        new_individual = individual
        
        # Calculate the expected improvement
        improvement = self.budget * np.exp(-self.explore_rate * np.abs(individual - np.array([5.0] * self.dim)))
        
        # If the improvement is greater than 0.2, refine the individual's strategy
        if improvement > 0.2 * np.random.rand():
            # Normalize the individual's fitness
            individual_fitness = np.mean(individual)
            
            # Update the individual's fitness
            new_individual_fitness = individual_fitness + np.random.normal(0, 0.1) * improvement
            
            # Update the individual's strategy
            new_individual = np.array([np.random.uniform(-5.0, 5.0) for _ in range(self.dim)])
        
        return new_individual

# Test the algorithm
budget = 100
dim = 10
func = np.sin
mgdalr = MGDALR(budget, dim)
individual = np.array([-5.0] * dim)
mgdalr(individual)
print(mgdalr.update_individual(individual))