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

class Individual:
    def __init__(self, dim, func):
        self.dim = dim
        self.func = func
        self.x = np.array([np.random.uniform(-5.0, 5.0) for _ in range(dim)])

    def __call__(self, self.logger):
        return self.func(self.x)

class Population:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.individuals = []

    def __call__(self, func):
        for _ in range(self.budget):
            individual = Individual(self.dim, func)
            self.individuals.append(individual)
            individual.logger = self.logger
            individual.x = np.array([np.random.uniform(-5.0, 5.0) for _ in range(self.dim)])
            individual.evaluate_fitness(individual.func)

class IndividualLogger:
    def __init__(self):
        self.fitness = None

    def evaluate_fitness(self, func):
        self.fitness = func(0)
        return self.fitness

class BBOB:
    def __init__(self, func, population):
        self.func = func
        self.population = population
        self.logger = IndividualLogger()

    def __call__(self):
        return self.population

# Description: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
# BBOB
# ```