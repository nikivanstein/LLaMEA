import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            # Generate a new individual using the current population
            new_individual = self.evaluate_fitness(self.population)
            
            # Refine the strategy using the probability 0.05
            if random.random() < 0.05:
                # Select a random individual from the current population
                selected_individual = random.choice(self.population)
                
                # Evaluate the fitness of the selected individual
                fitness = self.evaluate_fitness(selected_individual)
                
                # Refine the search space by swapping the selected individual with the new individual
                self.search_space = np.delete(self.search_space, 0, axis=0)
                self.search_space = np.vstack((self.search_space, selected_individual))
                
                # Update the population with the new individual
                self.population = np.vstack((self.population, new_individual))
            else:
                # Generate a new individual using the current population
                new_individual = self.evaluate_fitness(self.population)
                
                # Add the new individual to the population
                self.population = np.vstack((self.population, new_individual))
            
            # Evaluate the fitness of the new individual
            fitness = self.evaluate_fitness(new_individual)
            
            # Refine the search space by swapping the new individual with the updated population
            self.search_space = np.delete(self.search_space, 0, axis=0)
            self.search_space = np.vstack((self.search_space, new_individual))
            
            # Update the population with the new individual
            self.population = np.vstack((self.population, new_individual))
            
            # Check if the budget is reached
            if np.linalg.norm(self.func(new_individual)) < self.budget / 2:
                return new_individual