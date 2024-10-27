import numpy as np
import random
import operator

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

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

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.dim - 1)
            individual[idx] += np.random.uniform(-1, 1)
        
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            return np.concatenate((parent1[:int(self.dim / 2)], parent2[int(self.dim / 2):]))
        else:
            return np.concatenate((parent2[:int(self.dim / 2)], parent1[int(self.dim / 2):]))

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
        
        # Select the fittest individual
        fittest_individual = max(self.population, key=self.population.__getitem__)
        
        # Refine the strategy
        new_individual = fittest_individual
        for _ in range(self.population_size // 2):
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            new_individual = self.mutate(child)
        
        return new_individual

mgdalr = MGDALR(100, 10)
mgdalr.__call__(mgdalr)