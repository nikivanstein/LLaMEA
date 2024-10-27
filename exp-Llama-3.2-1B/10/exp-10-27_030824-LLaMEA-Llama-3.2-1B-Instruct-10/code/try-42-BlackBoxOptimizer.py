import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, mutation_rate=0.1, exploration_rate=0.1):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def optimize(self, func, mutation_rate=0.1, exploration_rate=0.1, max_iter=1000):
        # Initialize the population with random individuals
        population = [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(100)]
        
        # Evaluate the fitness of each individual in the population
        fitness = [self.__call__(func, mutation_rate, exploration_rate)[1] for individual in population]
        
        # Evolve the population over iterations
        for _ in range(max_iter):
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)]
            
            # Create a new generation by crossover and mutation
            new_population = []
            for _ in range(100):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = (parent1 + parent2) / 2
                if random.random() < exploration_rate:
                    child = random.uniform(self.search_space[0], self.search_space[1])
                new_population.append(child)
            
            # Evaluate the fitness of the new generation
            new_fitness = [self.__call__(func, mutation_rate, exploration_rate)[1] for individual in new_population]
            
            # Replace the old population with the new one
            population = new_population
            
            # Update the fitness of the fittest individuals
            fitness = new_fitness
        return population