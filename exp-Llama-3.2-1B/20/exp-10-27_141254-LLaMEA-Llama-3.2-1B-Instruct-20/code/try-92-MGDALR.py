import numpy as np
import random
import operator
from collections import deque

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

class BBOB:
    def __init__(self, func, budget, dim):
        self.func = func
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = deque(maxlen=self.population_size)
        self.population_list = []

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
        
        # Add the new individual to the population
        self.population.append(x)
        self.population_list.append(x)

    def run(self):
        # Run the evolutionary strategy
        for _ in range(self.budget):
            # Select the fittest individuals
            fittest_individuals = sorted(self.population, key=self.func, reverse=True)[:self.population_size // 2]
            
            # Select parents using tournament selection
            parents = []
            for _ in range(self.population_size // 2):
                individual = random.choice(fittest_individuals)
                tournament_size = random.randint(1, self.population_size // 2)
                winners = sorted(self.population, key=self.func, reverse=True)[:tournament_size]
                parent = random.choice(winners)
                parents.append(individual)
            
            # Crossover
            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + parent2) / 2
                offspring.append(child)
            
            # Mutate
            mutated_offspring = []
            for individual in offspring:
                if random.random() < 0.01:
                    mutation_rate = 0.01
                    mutated_individual = individual.copy()
                    mutated_individual[random.randint(0, self.dim - 1)] += random.uniform(-mutation_rate, mutation_rate)
                    mutated_offspring.append(mutated_individual)
            
            # Replace the least fit individuals with the new offspring
            self.population = deque(mutated_offspring + fittest_individuals)

# Select the fittest individual from the population
def select_fittest(individual, func):
    return func(individual)

# Evaluate the function at a given individual
def evaluate_fitness(individual, func):
    return func(individual)

# Run the evolutionary strategy
bboo = BBOB(func, budget, dim)
bboo.run()

# Update the population with the selected solution
def update_population(individual, func, budget):
    return select_fittest(individual, func), func(individual)

# Run the evolutionary strategy again
bboo = BBOB(func, budget, dim)
bboo.run()

# Update the population with the selected solution
def update_population(individual, func, budget):
    return select_fittest(individual, func), func(individual)

# Run the evolutionary strategy again
bboo = BBOB(func, budget, dim)
bboo.run()

# Update the population with the selected solution
def update_population(individual, func, budget):
    return select_fittest(individual, func), func(individual)

# Run the evolutionary strategy again
bboo = BBOB(func, budget, dim)
bboo.run()

# Update the population with the selected solution
def update_population(individual, func, budget):
    return select_fittest(individual, func), func(individual)

# Run the evolutionary strategy again
bboo = BBOB(func, budget, dim)
bboo.run()