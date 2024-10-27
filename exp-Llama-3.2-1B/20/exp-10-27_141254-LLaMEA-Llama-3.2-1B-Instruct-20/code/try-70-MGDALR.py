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
        self.explore_strategy = 'random'

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
            if self.explore_strategy == 'random':
                learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                dx = -np.dot(x - inner(x), np.gradient(y))
                x += learning_rate * dx
            elif self.explore_strategy == 'crossover':
                parent1 = x[:random.randint(0, self.dim)]
                parent2 = x[random.randint(0, self.dim)]
                child = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
                for i in range(self.dim):
                    child[i] = (parent1[i] + parent2[i]) / 2
                x = child
            elif self.explore_strategy =='mutation':
                learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                dx = -np.dot(x - inner(x), np.gradient(y))
                x += learning_rate * dx
                for i in range(self.dim):
                    if random.random() < 0.5:
                        x[i] += np.random.uniform(-0.1, 0.1)
            else:
                raise ValueError('Invalid exploration strategy')
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

class Individual:
    def __init__(self, dim):
        self.dim = dim
        self.fitness = 0

    def __call__(self, func):
        x = np.array([-5.0] * self.dim)
        for _ in range(func(self.f, x)):
            x = inner(x)
        return x

def inner(x):
    return np.sum(x**2)

def func(individual, x):
    return individual.f(x)

# Initialize the population
pop = [Individual(10) for _ in range(100)]

# Run the algorithm
for _ in range(100):
    algorithm = MGDALR(100, 10)
    for individual in pop:
        algorithm(individual)
        individual.fitness = func(individual, algorithm(x))
    print(f'Individual {individual} fitness: {individual.fitness}')

# Update the population
pop = [individual for individual in pop if individual.fitness > algorithm(pop[0]).fitness]