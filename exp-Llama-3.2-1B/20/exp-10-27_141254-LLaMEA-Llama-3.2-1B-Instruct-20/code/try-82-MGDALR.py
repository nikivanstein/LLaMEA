# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random

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

def gradient(func, x):
    return np.gradient(func(x))

def evaluate_fitness(individual, func, logger):
    updated_individual = individual
    for _ in range(100):
        updated_individual = func(updated_individual)
        logger.update(individual, updated_individual)
    return updated_individual

def mutation(individual, func, logger):
    new_individual = individual
    for _ in range(10):
        new_individual = func(new_individual)
        logger.update(individual, new_individual)
    return new_individual

def explore(individual, func, logger, budget):
    for _ in range(budget):
        func(individual)
        if random.random() < 0.2:
            individual = mutation(individual, func, logger, budget)
    return individual

class BBOB:
    def __init__(self):
        self.funcs = [
            lambda x: x**2,
            lambda x: 3*x**2 - 2*x + 1,
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: x**3 - 2*x**2 + x - 1,
            lambda x: x**4 - 2*x**3 + x**2 - x,
            lambda x: np.exp(x),
            lambda x: x**5 - 2*x**4 + x**3 - x**2 + x,
            lambda x: np.cos(x) + 2*x**2 + np.sin(x),
            lambda x: x**6 - 2*x**5 + x**4 - x**3 + x**2,
            lambda x: np.sin(x) + 2*x**2 - np.cos(x),
            lambda x: x**7 - 2*x**6 + x**5 - x**4 + x**3 - x**2 + x,
            lambda x: np.exp(x) + 2*x**2 - np.cos(x),
            lambda x: x**8 - 2*x**7 + x**6 - x**5 + x**4 - x**3 + x**2 - x,
        ]
        self.budget = 1000
        self.dim = 10
        self.logger = logger

    def run(self):
        individual = np.array([-5.0] * self.dim)
        best_individual = individual
        best_score = -np.inf
        for _ in range(self.budget):
            func = random.choice(self.funcs)
            individual = explore(individual, func, self.logger, self.budget)
            score = evaluate_fitness(individual, func, self.logger)
            if score > best_score:
                best_individual = individual
                best_score = score
        return best_individual, best_score

# Initialize the BBOB instance
bboo = BBOB()

# Run the optimization algorithm
best_individual, best_score = bboo.run()

# Print the results
print("Best Individual:", best_individual)
print("Best Score:", best_score)

# Update the MGDALR instance with the new best individual and score
mgdalr = MGDALR(bboo.budget, bboo.dim)
mgdalr.explore(best_individual, bboo.funcs[0], bboo.logger, bboo.budget)